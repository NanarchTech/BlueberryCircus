//! BlueberryCircus RK4 + Landau-Lifshitz SED integrator, C-ABI, dependency-free.
//!
//! This is a line-for-line port of the reference integrator in
//! `blueberry_circus/dynamics.py` (`_accel` + the RK4 step). The Python NumPy
//! backend remains the trust root; this Rust hot loop is diffed against it under
//! enclosure tolerance (1-ULP CPython/Rust divergence is expected, never
//! bit-equality). The boundary is pure data: f64 arrays in, f64 arrays out, no
//! objects cross.
//!
//! Field layout: `field` is `n_modes * 9` contiguous f64, each mode being
//! `[omega, kx, ky, kz, ex, ey, ez, amp, phase]` (a frozen ZPF realization,
//! identical to the one handed to NumPy so trajectories are comparable).

use std::slice;

#[inline]
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Potential force at x. kind 0 = harmonic (p0=omega0, p1=POTENTIAL mass), 1 =
/// coulomb (p0=coef=Z*k_e*q^2, p1=softening). The harmonic uses the potential's
/// own mass (matches Python potentials.Harmonic.force), NOT the particle mass.
#[inline]
fn pot_force(kind: i32, p0: f64, p1: f64, x: &[f64; 3]) -> [f64; 3] {
    match kind {
        0 => {
            let c = -p1 * p0 * p0; // p1 = potential mass
            [c * x[0], c * x[1], c * x[2]]
        }
        1 => {
            let rs = (dot(x, x) + p1 * p1).sqrt();
            let c = -p0 / (rs * rs * rs);
            [c * x[0], c * x[1], c * x[2]]
        }
        _ => [0.0, 0.0, 0.0],
    }
}

/// Jacobian-times-velocity for the Landau-Lifshitz reduction, `J(x) v`.
#[inline]
fn pot_jac_v(kind: i32, p0: f64, p1: f64, x: &[f64; 3], v: &[f64; 3]) -> [f64; 3] {
    match kind {
        0 => {
            let c = -p1 * p0 * p0; // p1 = potential mass
            [c * v[0], c * v[1], c * v[2]]
        }
        1 => {
            let rs = (dot(x, x) + p1 * p1).sqrt();
            let r3 = rs * rs * rs;
            let r5 = r3 * rs * rs;
            let xv = dot(x, v);
            [
                -p0 * (v[0] / r3 - 3.0 * x[0] * xv / r5),
                -p0 * (v[1] / r3 - 3.0 * x[1] * xv / r5),
                -p0 * (v[2] / r3 - 3.0 * x[2] * xv / r5),
            ]
        }
        _ => [0.0, 0.0, 0.0],
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn accel(
    x: &[f64; 3],
    v: &[f64; 3],
    t: f64,
    pot_kind: i32,
    p0: f64,
    p1: f64,
    charge: f64,
    mass: f64,
    field: &[f64],
    n_modes: usize,
    rr: i32,
    dipole: bool,
    tau: f64,
    c_light: f64,
) -> [f64; 3] {
    let r_eval = if dipole { [0.0, 0.0, 0.0] } else { *x };
    let mut f = if pot_kind >= 0 {
        pot_force(pot_kind, p0, p1, x)
    } else {
        [0.0, 0.0, 0.0]
    };

    let mut e_field = [0.0f64; 3];
    let mut dedt = [0.0f64; 3];
    if n_modes > 0 {
        let mut b_field = [0.0f64; 3];
        for m in 0..n_modes {
            let o = m * 9;
            let omega = field[o];
            let k = [field[o + 1], field[o + 2], field[o + 3]];
            let e = [field[o + 4], field[o + 5], field[o + 6]];
            let amp = field[o + 7];
            let phase = field[o + 8];
            let arg = dot(&k, &r_eval) - omega * t + phase;
            let ca = arg.cos();
            let sa = arg.sin();
            for i in 0..3 {
                e_field[i] += amp * e[i] * ca;
                dedt[i] += amp * omega * e[i] * sa;
            }
            if !dipole {
                let kn = dot(&k, &k).sqrt().max(1e-300);
                let khat = [k[0] / kn, k[1] / kn, k[2] / kn];
                let bv = cross(&khat, &e);
                for i in 0..3 {
                    b_field[i] += amp * (bv[i] / c_light) * ca;
                }
            }
        }
        for i in 0..3 {
            f[i] += charge * e_field[i];
        }
        if !dipole {
            let vxb = cross(v, &b_field);
            for i in 0..3 {
                f[i] += charge * vxb[i];
            }
        }
    }

    if rr == 1 {
        let mut dfdt = if pot_kind >= 0 {
            pot_jac_v(pot_kind, p0, p1, x, v)
        } else {
            [0.0, 0.0, 0.0]
        };
        if n_modes > 0 {
            for i in 0..3 {
                dfdt[i] += charge * dedt[i];
            }
        }
        for i in 0..3 {
            f[i] += tau * dfdt[i];
        }
    }

    [f[0] / mass, f[1] / mass, f[2] / mass]
}

/// Integrate the SED equation of motion with fixed-step RK4 (variable h per step,
/// read from `t_grid`). Writes `out_x` and `out_v` as `n*3` row-major f64.
///
/// # Safety
/// All pointers must be valid for the stated lengths; `out_x`/`out_v` writable
/// for `n*3` f64 each; `field` readable for `n_modes*9` f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bc_integrate(
    pot_kind: i32,
    p0: f64,
    p1: f64,
    charge: f64,
    mass: f64,
    n_modes: i64,
    field: *const f64,
    n: i64,
    t_grid: *const f64,
    x0: *const f64,
    v0: *const f64,
    rr: i32,
    dipole: i32,
    tau: f64,
    c_light: f64,
    out_x: *mut f64,
    out_v: *mut f64,
) {
    let nm = n_modes.max(0) as usize;
    let n = n.max(0) as usize;
    if n == 0 {
        return;
    }
    let field = if nm > 0 {
        slice::from_raw_parts(field, nm * 9)
    } else {
        &[]
    };
    let tg = slice::from_raw_parts(t_grid, n);
    let x0 = slice::from_raw_parts(x0, 3);
    let v0 = slice::from_raw_parts(v0, 3);
    let ox = slice::from_raw_parts_mut(out_x, n * 3);
    let ov = slice::from_raw_parts_mut(out_v, n * 3);
    let dip = dipole != 0;

    let mut x = [x0[0], x0[1], x0[2]];
    let mut v = [v0[0], v0[1], v0[2]];
    ox[0..3].copy_from_slice(&x);
    ov[0..3].copy_from_slice(&v);

    for i in 0..n - 1 {
        let t = tg[i];
        let h = tg[i + 1] - t;
        let a = |xx: &[f64; 3], vv: &[f64; 3], tt: f64| {
            accel(xx, vv, tt, pot_kind, p0, p1, charge, mass, field, nm, rr, dip, tau, c_light)
        };
        // k1
        let k1x = v;
        let k1v = a(&x, &v, t);
        // k2
        let x2 = [x[0] + 0.5 * h * k1x[0], x[1] + 0.5 * h * k1x[1], x[2] + 0.5 * h * k1x[2]];
        let v2in = [v[0] + 0.5 * h * k1v[0], v[1] + 0.5 * h * k1v[1], v[2] + 0.5 * h * k1v[2]];
        let k2x = v2in;
        let k2v = a(&x2, &v2in, t + 0.5 * h);
        // k3
        let x3 = [x[0] + 0.5 * h * k2x[0], x[1] + 0.5 * h * k2x[1], x[2] + 0.5 * h * k2x[2]];
        let v3in = [v[0] + 0.5 * h * k2v[0], v[1] + 0.5 * h * k2v[1], v[2] + 0.5 * h * k2v[2]];
        let k3x = v3in;
        let k3v = a(&x3, &v3in, t + 0.5 * h);
        // k4
        let x4 = [x[0] + h * k3x[0], x[1] + h * k3x[1], x[2] + h * k3x[2]];
        let v4in = [v[0] + h * k3v[0], v[1] + h * k3v[1], v[2] + h * k3v[2]];
        let k4x = v4in;
        let k4v = a(&x4, &v4in, t + h);

        for j in 0..3 {
            x[j] += (h / 6.0) * (k1x[j] + 2.0 * k2x[j] + 2.0 * k3x[j] + k4x[j]);
            v[j] += (h / 6.0) * (k1v[j] + 2.0 * k2v[j] + 2.0 * k3v[j] + k4v[j]);
        }
        let o = (i + 1) * 3;
        ox[o..o + 3].copy_from_slice(&x);
        ov[o..o + 3].copy_from_slice(&v);
    }
}
