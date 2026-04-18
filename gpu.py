import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import os
import csv
import warp as wp

wp.init()
wp.set_device("cuda")

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================
p_degree = 4
Np = p_degree + 1
N_per_elem = Np * Np # 25

# =============================================================================
# WARP KERNELS (Upgraded to float32)
# =============================================================================
@wp.kernel
def copy_h_col_kernel(h_col: wp.array(dtype=wp.float32), H: wp.array(dtype=wp.float32), k: wp.int32, max_iter: wp.int32):
    tid = wp.tid()
    if tid <= k + 1:
        H[tid * max_iter + k] = h_col[tid]

@wp.kernel
def apply_givens_kernel(H: wp.array(dtype=wp.float32), cs: wp.array(dtype=wp.float32), sn: wp.array(dtype=wp.float32), s: wp.array(dtype=wp.float32), k: wp.int32, max_iter: wp.int32):
    if wp.tid() == 0:
        for i in range(k):
            temp = cs[i] * H[i * max_iter + k] + sn[i] * H[(i + 1) * max_iter + k]
            H[(i + 1) * max_iter + k] = -sn[i] * H[i * max_iter + k] + cs[i] * H[(i + 1) * max_iter + k]
            H[i * max_iter + k] = temp

        denom = wp.sqrt(H[k * max_iter + k] * H[k * max_iter + k] + H[(k + 1) * max_iter + k] * H[(k + 1) * max_iter + k])
        if denom > wp.float32(1e-6):
            cs[k] = H[k * max_iter + k] / denom
            sn[k] = H[(k + 1) * max_iter + k] / denom
        else:
            cs[k] = wp.float32(1.0)
            sn[k] = wp.float32(0.0)

        H[k * max_iter + k] = cs[k] * H[k * max_iter + k] + sn[k] * H[(k + 1) * max_iter + k]
        H[(k + 1) * max_iter + k] = wp.float32(0.0)

        temp_s = cs[k] * s[k] + sn[k] * s[k + 1]
        s[k + 1] = -sn[k] * s[k] + cs[k] * s[k + 1]
        s[k] = temp_s

@wp.kernel
def backward_substitution_kernel(H: wp.array(dtype=wp.float32), s: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), k: wp.int32, max_iter: wp.int32):
    if wp.tid() == 0:
        for i in range(k - 1, -1, -1):
            y[i] = s[i]
            for j in range(i + 1, k):
                y[i] = y[i] - H[i * max_iter + j] * y[j]
            y[i] = y[i] / H[i * max_iter + i]

@wp.kernel
def axpy_from_y_vec_kernel(y_vec: wp.array(dtype=wp.float32), V_k: wp.array(dtype=wp.float32), u_guess: wp.array(dtype=wp.float32), k: wp.int32):
    tid = wp.tid()
    if tid < u_guess.shape[0]:
        u_guess[tid] = u_guess[tid] + y_vec[k] * V_k[tid]
@wp.func
def get_primitives_wp(n: wp.float32, nE: wp.float32, gx: wp.float32, gy: wp.float32, vg2: wp.float32):
    g_norm = wp.sqrt(gx*gx + gy*gy)
    
    # Cast literal to float32 to prevent wp.max() type mismatch
    nE_safe = wp.max(nE, wp.float32(1e-6))
    
    u_norm = wp.float32(0.0)
    if g_norm > wp.float32(1e-6):
        disc = wp.float32(9.0) * nE_safe * nE_safe - wp.float32(8.0) * g_norm * g_norm * vg2
        u_norm = (wp.float32(3.0) * nE_safe - wp.sqrt(wp.max(disc, wp.float32(0.0)))) / (wp.float32(2.0) * g_norm)
    else:
        u_norm = (wp.float32(2.0) * g_norm * vg2) / (wp.float32(3.0) * nE_safe)
        
    u_norm = wp.min(u_norm, wp.float32(0.999) * wp.sqrt(vg2))
    v2 = (u_norm * u_norm) / vg2
    W = wp.float32(3.0) * nE_safe / (wp.float32(2.0) + v2 + wp.float32(1e-6))
    P = nE_safe * (wp.float32(1.0) - v2) / (wp.float32(2.0) + v2 + wp.float32(1e-6))
    
    ux, uy = wp.float32(0.0), wp.float32(0.0)
    if g_norm > wp.float32(1e-6):
        ux = (gx / g_norm) * u_norm 
        uy = (gy / g_norm) * u_norm
    return ux, uy, P, W

@wp.func
def get_transport_wp(mu: wp.float32, B: wp.float32, vg2: wp.float32):
    vg = wp.sqrt(vg2)
    tau_ee = wp.float32(0.1)
    tau_dis = wp.float32(10.0)
    
    # Mathematically, abs(sign(mu) * mu^2) == mu^2. 
    # This bypasses wp.sign() and keeps precision strictly at float32.
    n_abs = mu * mu 
    
    W = wp.float32(1.5) * wp.pow(n_abs, wp.float32(1.5))
    W_safe = wp.max(W, wp.float32(1e-6))
    
    eta_0 = tau_ee * vg * vg * W_safe * wp.float32(0.25)
    omega_c = B * vg * vg / W_safe
    
    two_om_tau = wp.float32(2.0) * omega_c * tau_ee
    denom = wp.float32(1.0) + (two_om_tau * two_om_tau)
    
    return eta_0 / denom, (eta_0 * two_om_tau) / denom, tau_dis

@wp.kernel
def matvec_viscosity_fused_kernel(
    x: wp.array(dtype=wp.float32),       
    y: wp.array(dtype=wp.float32),       
    U_base: wp.array(dtype=wp.float32),  
    M_local: wp.array(dtype=wp.float32),
    Dx_phys: wp.array(dtype=wp.float32),
    Dy_phys: wp.array(dtype=wp.float32),
    D1d_phys_x: wp.array(dtype=wp.float32),
    D1d_phys_y: wp.array(dtype=wp.float32),
    nx: wp.int32, ny: wp.int32, Ne: wp.int32,
    B_field: wp.float32, vg2: wp.float32, dt_gamma: wp.float32,
    hx: wp.float32, hy: wp.float32, slip_length: wp.float32,
    qpc_faces: wp.array(dtype=wp.int32),
    qpc_centers: wp.array(dtype=wp.float32),
    qpc_widths: wp.array(dtype=wp.float32),
    face_nodes: wp.array(dtype=wp.int32, ndim=2),
    normals_vals: wp.array(dtype=wp.float32, ndim=2),
    weights1d: wp.array(dtype=wp.float32)
):
    e = wp.tid()
    if e >= Ne: 
        return

    local_x_gx = wp.zeros(shape=25, dtype=wp.float32)
    local_x_gy = wp.zeros(shape=25, dtype=wp.float32)
    local_u_diff_x = wp.zeros(shape=25, dtype=wp.float32)
    local_u_diff_y = wp.zeros(shape=25, dtype=wp.float32)

    for i in range(25):
        idx_base = e * 25 + i
        y[idx_base] = x[idx_base] * M_local[i]
        y[25 * Ne + idx_base] = x[25 * Ne + idx_base] * M_local[i]
        local_x_gx[i] = x[50 * Ne + idx_base]
        local_x_gy[i] = x[75 * Ne + idx_base]
        
        y[50 * Ne + idx_base] = local_x_gx[i] * M_local[i]
        y[75 * Ne + idx_base] = local_x_gy[i] * M_local[i]

    tau_sum = wp.float32(0.0)
    eta_local = wp.zeros(shape=25, dtype=wp.float32)
    etaH_local = wp.zeros(shape=25, dtype=wp.float32) 
    for i in range(25):
        idx = e * 25 + i
        n_val = U_base[idx]
        nE_val = U_base[25 * Ne + idx]
        gx_val = U_base[50 * Ne + idx]
        gy_val = U_base[75 * Ne + idx]

        mu = wp.sqrt(wp.abs(n_val)) * wp.sign(n_val)
        eta, etaH, tau = get_transport_wp(mu, B_field, vg2)
        eta_local[i] = eta
        etaH_local[i] = etaH
        tau_sum += tau
        
        _, _, _, W = get_primitives_wp(n_val, nE_val, gx_val, gy_val, vg2)
        local_u_diff_x[i] = local_x_gx[i] * vg2 / wp.max(W, wp.float32(1e-6))
        local_u_diff_y[i] = local_x_gy[i] * vg2 / wp.max(W, wp.float32(1e-6))

    # ADDED: Safety floor for tau_avg division
    tau_avg = tau_sum / wp.float32(25.0)
    tau_avg_safe = wp.max(tau_avg, wp.float32(1e-6))
    
    gux_x = wp.zeros(shape=25, dtype=wp.float32)
    gux_y = wp.zeros(shape=25, dtype=wp.float32)
    guy_x = wp.zeros(shape=25, dtype=wp.float32)
    guy_y = wp.zeros(shape=25, dtype=wp.float32)
    # D1d_gpu must be passed as an argument: wp.array(dtype=wp.float32, shape=(5,5))
    for iy in range(5):
        for ix in range(5):
            i = iy * 5 + ix
            gx_x_acc, gx_y_acc, gy_x_acc, gy_y_acc = wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
            
            # Sum Factorization: D_x applies along x, D_y applies along y
            for k in range(5):
                # Apply Dx (operates on x-indices)
                idx_x = iy * 5 + k
                D_val_x = D1d_phys_x[ix * 5 + k]
                gx_x_acc += local_u_diff_x[idx_x] * D_val_x
                gy_x_acc += local_u_diff_y[idx_x] * D_val_x
                
                # Apply Dy (operates on y-indices)
                idx_y = k * 5 + ix
                D_val_y = D1d_phys_y[iy * 5 + k]
                gx_y_acc += local_u_diff_x[idx_y] * D_val_y
                gy_y_acc += local_u_diff_y[idx_y] * D_val_y

            gux_x[i] = gx_x_acc
            gux_y[i] = gx_y_acc
            guy_x[i] = gy_x_acc
            guy_y[i] = gy_y_acc
    for i in range(25):
        tgx = wp.float32(0.0)
        tgy = wp.float32(0.0)
        for j in range(25):
            v_eta_x = eta_local[j] * (gux_x[j] * Dx_phys[j * 25 + i] + gux_y[j] * Dy_phys[j * 25 + i])
            v_etaH_x = etaH_local[j] * (gux_x[j] * Dy_phys[j * 25 + i] - gux_y[j] * Dx_phys[j * 25 + i])

            v_eta_y = eta_local[j] * (guy_x[j] * Dx_phys[j * 25 + i] + guy_y[j] * Dy_phys[j * 25 + i])
            v_etaH_y = etaH_local[j] * (-guy_x[j] * Dy_phys[j * 25 + i] + guy_y[j] * Dx_phys[j * 25 + i])

            # Couple x and y spatial gradients for Hall viscosity (transverse force)
            tgx += M_local[j] * (v_eta_x - v_etaH_y)
            tgy += M_local[j] * (v_eta_y + v_etaH_x)
        
        idx = e * 25 + i
        y[50 * Ne + idx] += dt_gamma * (vg2 * tgx + (local_x_gx[i] / tau_avg_safe) * M_local[i])
        y[75 * Ne + idx] += dt_gamma * (vg2 * tgy + (local_x_gy[i] / tau_avg_safe) * M_local[i])

    ex = e % nx
    ey = e // nx
    penalty_param = wp.float32(40.0 * 16.0) / wp.min(hx, hy)  # p_degree = 4 -> 4^2 = 16.0
    
    for face in range(4):
        nx_f = normals_vals[face, 0]
        ny_f = normals_vals[face, 1]
        
        # CORRECTED: Properly isolate Physical Boundaries from Interior Iterations
        is_boundary = int(0)
        if (face == 0 and ey == 0) or (face == 2 and ey == ny - 1) or (face == 3 and ex == 0) or (face == 1 and ex == nx - 1):
            is_boundary = int(1)
            
        is_wall = int(0)
        if is_boundary == 1:
            is_wall = int(1)
            for qidx in range(qpc_faces.shape[0]):
                if qpc_faces[qidx] == face:
                    emin = wp.float32(ex) / wp.float32(nx) if (face == 0 or face == 2) else wp.float32(ey) / wp.float32(ny)
                    emax = wp.float32(ex + 1) / wp.float32(nx) if (face == 0 or face == 2) else wp.float32(ey + 1) / wp.float32(ny)
                    if emax > qpc_centers[qidx] - wp.float32(0.5) * qpc_widths[qidx] and emin < qpc_centers[qidx] + wp.float32(0.5) * qpc_widths[qidx]:
                        is_wall = int(0)
                        break

        dsw = hx * wp.float32(0.5) if (face == 0 or face == 2) else hy * wp.float32(0.5)
        
        if is_boundary == 1:
            if is_wall == 1:
                mean_eta = wp.float32(0.0)
                for k in range(25):
                    mean_eta += eta_local[k]
                mean_eta /= wp.float32(25.0)
                wall_coeff = mean_eta / wp.max(slip_length, wp.float32(1e-6))
                
                for q in range(5):
                    idx_node = face_nodes[face, q]
                    ut = -local_u_diff_x[idx_node] * ny_f + local_u_diff_y[idx_node] * nx_f
                    wq = weights1d[q] * dsw
                    loc = e * 25 + idx_node
                    y[50 * Ne + loc] += dt_gamma * (vg2 * wall_coeff * ut * (-ny_f) * wq)
                    y[75 * Ne + loc] += dt_gamma * (vg2 * wall_coeff * ut * (nx_f) * wq)
            # Implicit else: continue (for QPC openings)
            
        else:
            # CORRECTED: INTERIOR FACE SIPG TERM 
            e2 = e - nx if face == 0 else e + 1 if face == 1 else e + nx if face == 2 else e - 1
            neighbor_face = int(2) if face == 0 else int(3) if face == 1 else int(0) if face == 2 else int(1)

            # Precompute neighboring primitive vectors to allow inline gradient derivation locally
            u2_x_local = wp.zeros(shape=25, dtype=wp.float32)
            u2_y_local = wp.zeros(shape=25, dtype=wp.float32)
            for k in range(25):
                idx_base2 = e2 * 25 + k
                n_val2 = U_base[idx_base2]
                nE_val2 = U_base[25 * Ne + idx_base2]
                gx_val2 = U_base[50 * Ne + idx_base2]
                gy_val2 = U_base[75 * Ne + idx_base2]
                x_gx2 = x[50 * Ne + idx_base2]
                x_gy2 = x[75 * Ne + idx_base2]
                
                _, _, _, W2 = get_primitives_wp(n_val2, nE_val2, gx_val2, gy_val2, vg2)
                u2_x_local[k] = x_gx2 * vg2 / wp.max(W2, wp.float32(1e-6))
                u2_y_local[k] = x_gy2 * vg2 / wp.max(W2, wp.float32(1e-6))

            for q in range(5):
                idx1 = face_nodes[face, q]
                idx2 = face_nodes[neighbor_face, q]
                wq = weights1d[q] * dsw

                u1_x = local_u_diff_x[idx1]
                u1_y = local_u_diff_y[idx1]
                u2_x = u2_x_local[idx2]
                u2_y = u2_y_local[idx2]

                eta1 = eta_local[idx1]
                etaH1 = etaH_local[idx1]

                n_val2_q = U_base[e2 * 25 + idx2]
                mu2 = wp.sqrt(wp.abs(n_val2_q)) * wp.sign(n_val2_q)
                eta2, etaH2, _ = get_transport_wp(mu2, B_field, vg2)

                eta_avg = wp.float32(0.5) * (eta1 + eta2)
                etaH_avg = wp.float32(0.5) * (etaH1 + etaH2)

                gn_u1_x = gux_x[idx1] * nx_f + gux_y[idx1] * ny_f
                gn_u1_y = guy_x[idx1] * nx_f + guy_y[idx1] * ny_f

                gn_u2_x = wp.float32(0.0)
                gn_u2_y = wp.float32(0.0)
                for k in range(25):
                    gn_u2_x += u2_x_local[k] * (Dx_phys[idx2 * 25 + k] * nx_f + Dy_phys[idx2 * 25 + k] * ny_f)
                    gn_u2_y += u2_y_local[k] * (Dx_phys[idx2 * 25 + k] * nx_f + Dy_phys[idx2 * 25 + k] * ny_f)

                q1x = eta1 * gn_u1_x - etaH1 * gn_u1_y
                q1y = etaH1 * gn_u1_x + eta1 * gn_u1_y

                q2x = eta2 * gn_u2_x - etaH2 * gn_u2_y
                q2y = etaH2 * gn_u2_x + eta2 * gn_u2_y

                q_avg_x = wp.float32(0.5) * (q1x + q2x)
                q_avg_y = wp.float32(0.5) * (q1y + q2y)

                for i in range(25):
                    v1 = wp.float32(1.0) if i == idx1 else wp.float32(0.0)
                    gn_v1 = Dx_phys[idx1 * 25 + i] * nx_f + Dy_phys[idx1 * 25 + i] * ny_f

                    # Corrected signs to match cpu.py anti-symmetric tensor
                    flux_x = (-q_avg_x * v1) + \
                    (-wp.float32(0.5) * gn_v1 * (eta_avg * (u1_x - u2_x) - etaH_avg * (u1_y - u2_y))) + \
                    (penalty_param * eta_avg * (u1_x - u2_x) * v1)

                    flux_y = (-q_avg_y * v1) + \
                    (-wp.float32(0.5) * gn_v1 * (etaH_avg * (u1_x - u2_x) + eta_avg * (u1_y - u2_y))) + \
                    (penalty_param * eta_avg * (u1_y - u2_y) * v1)
                    loc = e * 25 + i
                    y[50 * Ne + loc] += dt_gamma * vg2 * flux_x * wq
                    y[75 * Ne + loc] += dt_gamma * vg2 * flux_y * wq

"""
Calculates the ideal, convective portion of the equations including
thermodynamic pressure gradients and convective derivatives.
Uses a local Lax-Friedrichs flux for stable shock capturing at boundaries.
"""
@wp.kernel
def compute_explicit_euler_flux_kernel(
    U: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    M_local: wp.array(dtype=wp.float32),
    Dx_phys: wp.array(dtype=wp.float32),
    Dy_phys: wp.array(dtype=wp.float32),
    nx: wp.int32, ny: wp.int32, Ne: wp.int32,
    vg2: wp.float32,
    n_src: wp.float32, nE_src: wp.float32, n_drn: wp.float32, nE_drn: wp.float32,
    qpc_faces: wp.array(dtype=wp.int32),
    qpc_centers: wp.array(dtype=wp.float32),
    qpc_widths: wp.array(dtype=wp.float32),
    qpc_types: wp.array(dtype=wp.int32),
    face_nodes: wp.array(dtype=wp.int32, ndim=2),
    normals_vals: wp.array(dtype=wp.float32, ndim=2),
    weights1d: wp.array(dtype=wp.float32),
    hx: wp.float32, hy: wp.float32
):
    e = wp.tid()
    if e >= Ne: 
        return

    local_n = wp.zeros(shape=25, dtype=wp.float32)
    local_nE = wp.zeros(shape=25, dtype=wp.float32)
    local_gx = wp.zeros(shape=25, dtype=wp.float32)
    local_gy = wp.zeros(shape=25, dtype=wp.float32)
    
    local_ux = wp.zeros(shape=25, dtype=wp.float32)
    local_uy = wp.zeros(shape=25, dtype=wp.float32)
    local_P = wp.zeros(shape=25, dtype=wp.float32)
    local_W = wp.zeros(shape=25, dtype=wp.float32)
    for i in range(25):
        idx = e * 25 + i
        local_n[i] = U[idx]
        local_nE[i] = U[25 * Ne + idx]
        local_gx[i] = U[50 * Ne + idx]
        local_gy[i] = U[75 * Ne + idx]
        
        ux, uy, P, W = get_primitives_wp(local_n[i], local_nE[i], local_gx[i], local_gy[i], vg2)
        local_ux[i] = ux
        local_uy[i] = uy
        local_P[i] = P
        local_W[i] = W

    fn_x = wp.zeros(shape=25, dtype=wp.float32)
    fn_y = wp.zeros(shape=25, dtype=wp.float32)
    fnE_x = wp.zeros(shape=25, dtype=wp.float32)
    fnE_y = wp.zeros(shape=25, dtype=wp.float32)
    fgx_x = wp.zeros(shape=25, dtype=wp.float32)
    fgx_y = wp.zeros(shape=25, dtype=wp.float32)
    fgy_x = wp.zeros(shape=25, dtype=wp.float32)
    fgy_y = wp.zeros(shape=25, dtype=wp.float32)

    for i in range(25):
        fn_x[i] = local_n[i] * local_ux[i]
        fn_y[i] = local_n[i] * local_uy[i]
        fnE_x[i] = local_W[i] * local_ux[i]
        fnE_y[i] = local_W[i] * local_uy[i]
        fgx_x[i] = local_ux[i] * local_gx[i] + vg2 * local_P[i]
        fgx_y[i] = local_uy[i] * local_gx[i]
        fgy_x[i] = local_ux[i] * local_gy[i]
        fgy_y[i] = local_uy[i] * local_gy[i] + vg2 * local_P[i]

    for i in range(25):
        rn_acc = wp.float32(0.0)
        rnE_acc = wp.float32(0.0)
        rgx_acc = wp.float32(0.0)
        rgy_acc = wp.float32(0.0)
        for j in range(25):
            rn_acc += fn_x[j] * Dx_phys[i * 25 + j] + fn_y[j] * Dy_phys[i * 25 + j]
            rnE_acc += fnE_x[j] * Dx_phys[i * 25 + j] + fnE_y[j] * Dy_phys[i * 25 + j]
            rgx_acc += fgx_x[j] * Dx_phys[i * 25 + j] + fgx_y[j] * Dy_phys[i * 25 + j]
            rgy_acc += fgy_x[j] * Dx_phys[i * 25 + j] + fgy_y[j] * Dy_phys[i * 25 + j]
        
        idx = e * 25 + i
        rhs[idx] = -rn_acc * M_local[i]
        rhs[25 * Ne + idx] = -rnE_acc * M_local[i]
        rhs[50 * Ne + idx] = -rgx_acc * M_local[i]
        rhs[75 * Ne + idx] = -rgy_acc * M_local[i]

    ex = e % nx
    ey = e // nx
    for face in range(4):
        nx_f = normals_vals[face, 0]
        ny_f = normals_vals[face, 1]
        
        is_boundary = int(0)
        if (face == 0 and ey == 0) or (face == 2 and ey == ny - 1) or (face == 3 and ex == 0) or (face == 1 and ex == nx - 1):
            is_boundary = int(1)
        e2 = e
        b_type = int(0)
        if is_boundary == 1:
            for q_idx in range(qpc_faces.shape[0]):
                if qpc_faces[q_idx] == face:
                    emin = wp.float32(ex) / wp.float32(nx) if (face == 0 or face == 2) else wp.float32(ey) / wp.float32(ny)
                    emax = wp.float32(ex + 1) / wp.float32(nx) if (face == 0 or face == 2) else wp.float32(ey + 1) / wp.float32(ny)
                    if emax > qpc_centers[q_idx] - wp.float32(0.5) * qpc_widths[q_idx] and emin < qpc_centers[q_idx] + wp.float32(0.5) * qpc_widths[q_idx]:
                        b_type = int(1) if qpc_types[q_idx] == 0 else int(2)
                        break
        else:
            e2 = (e - nx if face == 0 else e + 1 if face == 1 else e + nx if face == 2 else e - 1)

        ds_w_base = hx * wp.float32(0.5) if (face == 0 or face == 2) else hy * wp.float32(0.5)
        
        for q in range(5):
            idx1 = face_nodes[face, q]
            wq = weights1d[q] * ds_w_base
            
            n1 = local_n[idx1]
            nE1 = local_nE[idx1]
            gx1 = local_gx[idx1]
            gy1 = local_gy[idx1]
            ux1, uy1, P1, W1 = get_primitives_wp(n1, nE1, gx1, gy1, vg2)
            
            n2, nE2, gx2, gy2 = wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
            if is_boundary == 1:
                if b_type == 1:
                    n2, nE2, gx2, gy2 = n_src, nE_src, gx1, gy1
                elif b_type == 2:
                    n2, nE2, gx2, gy2 = n_drn, nE_drn, gx1, gy1
                else: 
                    un1 = ux1 * nx_f + uy1 * ny_f
                    ut1 = -ux1 * ny_f + uy1 * nx_f
                    ux2 = -un1 * nx_f - ut1 * ny_f
                    uy2 = -un1 * ny_f + ut1 * nx_f
                    gn1 = gx1 * nx_f + gy1 * ny_f
                    gt1 = -gx1 * ny_f + gy1 * nx_f
                    gx2 = -gn1 * nx_f - gt1 * ny_f
                    gy2 = -gn1 * ny_f + gt1 * nx_f
                    n2, nE2 = n1, nE1
            else:
                neighbor_face = int(2) if face == 0 else int(3) if face == 1 else int(0) if face == 2 else int(1)
                idx2 = face_nodes[neighbor_face, q]
                
                idx2_base = e2 * 25 + idx2 
                n2 = U[idx2_base]
                nE2 = U[25 * Ne + idx2_base]
                gx2 = U[50 * Ne + idx2_base]
                gy2 = U[75 * Ne + idx2_base]
            
            ux2, uy2, P2, W2 = get_primitives_wp(n2, nE2, gx2, gy2, vg2)
            
            un1 = ux1 * nx_f + uy1 * ny_f
            un2 = ux2 * nx_f + uy2 * ny_f
            
            f1n = n1 * un1
            f2n = n2 * un2
            f1nE = W1 * un1
            f2nE = W2 * un2
            f1gx = (ux1 * gx1 + vg2 * P1) * nx_f + (uy1 * gx1) * ny_f
            f2gx = (ux2 * gx2 + vg2 * P2) * nx_f + (uy2 * gx2) * ny_f
            f1gy = (ux1 * gy1) * nx_f + (uy1 * gy1 + vg2 * P1) * ny_f
            f2gy = (ux2 * gy2) * nx_f + (uy2 * gy2 + vg2 * P2) * ny_f
            
            cs1 = wp.sqrt(vg2 * (wp.float32(1.0) - (ux1 * ux1 + uy1 * uy1) / vg2) / (wp.float32(2.0) + (ux1 * ux1 + uy1 * uy1) / vg2))
            cs2 = wp.sqrt(vg2 * (wp.float32(1.0) - (ux2 * ux2 + uy2 * uy2) / vg2) / (wp.float32(2.0) + (ux2 * ux2 + uy2 * uy2) / vg2))
            lam = wp.max(wp.abs(un1) + cs1, wp.abs(un2) + cs2) + wp.float32(1e-6)
            
            half = wp.float32(0.5)
            loc = e * 25 + idx1
            rhs[loc] -= (half * (f1n + f2n) - half * lam * (n2 - n1) - f1n) * wq
            rhs[25 * Ne + loc] -= (half * (f1nE + f2nE) - half * lam * (nE2 - nE1) - f1nE) * wq
            rhs[50 * Ne + loc] -= (half * (f1gx + f2gx) - half * lam * (gx2 - gx1) - f1gx) * wq
            rhs[75 * Ne + loc] -= (half * (f1gy + f2gy) - half * lam * (gy2 - gy1) - f1gy) * wq

"""
Applies external electromagnetic forces algebraically.
Adds macroscopic Lorentz force to the momentum continuity equation,
and Joule heating dissipation to the energy density equation.
"""
@wp.kernel
def compute_electromagnetic_source_kernel(
    U: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    M_local: wp.array(dtype=wp.float32),
    Ex_full: wp.array(dtype=wp.float32),
    Ey_full: wp.array(dtype=wp.float32),
    Ne: wp.int32,
    B_field: wp.float32, vg2: wp.float32, e_charge: wp.float32, c_light: wp.float32
):
    e = wp.tid()
    if e >= Ne:
        return

    for i in range(25):
        idx = e * 25 + i
        local_n = U[idx]
        local_nE = U[25 * Ne + idx]
        local_gx = U[50 * Ne + idx]
        local_gy = U[75 * Ne + idx]

        ux, uy, _, _ = get_primitives_wp(local_n, local_nE, local_gx, local_gy, vg2)

        Ex = Ex_full[idx]
        Ey = Ey_full[idx]
        jx = local_n * ux
        jy = local_n * uy

        rhs[25 * Ne + idx] += e_charge * (Ex * jx + Ey * jy) * M_local[i]
        rhs[50 * Ne + idx] += (e_charge * local_n * Ex + (e_charge / c_light) * jy * B_field) * M_local[i]
        rhs[75 * Ne + idx] += (e_charge * local_n * Ey - (e_charge / c_light) * jx * B_field) * M_local[i]

"""
Enforces sub-relativistic velocity limits to prevent imaginary numbers
when computing primitive variables. Applied to the intermediate state 'b'
before it enters the GMRES solver.
"""
@wp.kernel
def clamp_b_state_kernel(
    b: wp.array(dtype=wp.float32),
    Ne: wp.int32,
    vg2: wp.float32
):
    e = wp.tid()
    if e >= Ne:
        return

    for i in range(25):
        idx = e * 25 + i
        n_val = b[idx]
        nE_val = b[25 * Ne + idx]
        gx_val = b[50 * Ne + idx]
        gy_val = b[75 * Ne + idx]

        g_norm = wp.sqrt(gx_val * gx_val + gy_val * gy_val)
        nE_safe = wp.max(nE_val, wp.float32(1e-6))

        u_norm = wp.float32(0.0)
        if g_norm > wp.float32(1e-6):
            disc = wp.float32(9.0) * nE_safe * nE_safe - wp.float32(8.0) * g_norm * g_norm * vg2
            u_norm = (wp.float32(3.0) * nE_safe - wp.sqrt(wp.max(disc, wp.float32(0.0)))) / (wp.float32(2.0) * g_norm)
        else:
            u_norm = (wp.float32(2.0) * g_norm * vg2) / (wp.float32(3.0) * nE_safe)

        limit = wp.float32(0.999) * wp.sqrt(vg2)
        if u_norm > limit:
            scale = limit / u_norm
            b[50 * Ne + idx] = gx_val * scale
            b[75 * Ne + idx] = gy_val * scale

@wp.kernel
def compute_b_stage1_kernel(
    U_n: wp.array(dtype=wp.float32), 
    F_exp: wp.array(dtype=wp.float32), 
    M_local: wp.array(dtype=wp.float32), 
    dt_gamma: wp.float32, 
    b: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    if tid < b.shape[0]:
        b[tid] = U_n[tid] * M_local[tid % 25] + dt_gamma * F_exp[tid]

@wp.kernel
def compute_b_stage2_kernel(
    U_n: wp.array(dtype=wp.float32), 
    U_1: wp.array(dtype=wp.float32),
    F_exp_1: wp.array(dtype=wp.float32), 
    M_local: wp.array(dtype=wp.float32),
    dt_gamma2: wp.float32, 
    c_n: wp.float32, 
    c_1: wp.float32, 
    b: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    if tid < b.shape[0]:
        b[tid] = dt_gamma2 * F_exp_1[tid] + (c_n * U_n[tid] + c_1 * U_1[tid]) * M_local[tid % 25]

@wp.kernel
def dot_product_kernel(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    sum_val = wp.float32(0.0)
    while tid < a.shape[0]:
        sum_val += a[tid] * b[tid]
        tid += 1024
    wp.atomic_add(out, 0, sum_val)

@wp.kernel
def axpy_kernel(alpha: wp.float32, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if tid < x.shape[0]:
        y[tid] = alpha * x[tid] + y[tid]

@wp.kernel
def scale_kernel(x: wp.array(dtype=wp.float32), scale: wp.float32):
    tid = wp.tid()
    if tid < x.shape[0]:
        x[tid] = x[tid] * scale

@wp.kernel
def apply_preconditioner_kernel(x: wp.array(dtype=wp.float32), M_local: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if tid < x.shape[0]:
        x[tid] = x[tid] / M_local[tid % 25]

@wp.kernel
def axpy_from_scalar_buf_kernel(
    alpha_buf: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    if tid < x.shape[0]:
        y[tid] = y[tid] - alpha_buf[0] * x[tid]


@wp.kernel
def write_scalar_to_vec_kernel(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
    idx: wp.int32
):
    if wp.tid() == 0:
        dst[idx] = src[0]


@wp.kernel
def write_sqrt_scalar_to_vec_kernel(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
    idx: wp.int32
):
    if wp.tid() == 0:
        dst[idx] = wp.sqrt(wp.max(src[0], wp.float32(0.0)))


@wp.kernel
def normalize_from_dot_kernel(
    x: wp.array(dtype=wp.float32),
    dot_buf: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    if tid < x.shape[0]:
        inv_norm = wp.float32(1.0) / wp.sqrt(wp.max(dot_buf[0], wp.float32(1e-30)))
        x[tid] = x[tid] * inv_norm
# =============================================================================
# GPU GMRES SOLVER
# =============================================================================
class WarpGMRES:
     def __init__(self, nx, ny, max_iter=12, check_every=1):
        self.nx, self.ny = nx, ny
        self.Ne = nx * ny
        self.N = self.Ne * 25
        self.total_size = 4 * self.N
        self.max_iter = max_iter
        self.check_every = check_every

        self.V = [wp.zeros(self.total_size, dtype=wp.float32) for _ in range(self.max_iter + 1)]

        self.norm_buf = wp.zeros(1, dtype=wp.float32)
        self.dot_buf = wp.zeros(1, dtype=wp.float32)

        self.r = wp.zeros(self.total_size, dtype=wp.float32)
        self.w = wp.zeros(self.total_size, dtype=wp.float32)
        self.z = wp.zeros(self.total_size, dtype=wp.float32)

        # One Hessenberg column on GPU, mirrored once per Arnoldi step to CPU
        self.h_col_gpu = wp.zeros(self.max_iter + 1, dtype=wp.float32)
        self.h_col_cpu = wp.zeros(self.max_iter + 1, dtype=wp.float32, device="cpu")

        # Reusable scalar mirror on CPU
        self.scalar_cpu = wp.zeros(1, dtype=wp.float32, device="cpu")   
        self.H_gpu = wp.zeros((self.max_iter + 1) * self.max_iter, dtype=wp.float32)
        self.cs_gpu = wp.zeros(self.max_iter, dtype=wp.float32)
        self.sn_gpu = wp.zeros(self.max_iter, dtype=wp.float32)
        self.s_gpu = wp.zeros(self.max_iter + 1, dtype=wp.float32)
        self.y_gpu = wp.zeros(self.max_iter, dtype=wp.float32) 
     def solve(
        self,
        U_guess,
        b_gpu,
        dt_gamma,
        U_base_gpu,
        M_local_gpu,
        Dx_phys_gpu,
        Dy_phys_gpu,
        D1d_phys_x_gpu,
        D1d_phys_y_gpu,
        B_field,
        vg2,
        hx,
        hy,
        slip_length,
        qpc_f_wp,
        qpc_c_wp,
        qpc_w_wp,
        f_nodes_wp,
        norm_v_wp,
        w1d_wp
    ):
        self.r.zero_()
        wp.launch(
            kernel=matvec_viscosity_fused_kernel,
            dim=self.Ne,
            inputs=[
                U_guess, self.r, U_base_gpu, M_local_gpu, Dx_phys_gpu, Dy_phys_gpu, D1d_phys_x_gpu, D1d_phys_y_gpu,
                self.nx, self.ny, self.Ne, B_field, vg2, dt_gamma, hx, hy,
                slip_length, qpc_f_wp, qpc_c_wp, qpc_w_wp, f_nodes_wp, norm_v_wp, w1d_wp
            ]
        )

        wp.launch(kernel=scale_kernel, dim=self.total_size, inputs=[self.r, -1.0])
        wp.launch(kernel=axpy_kernel, dim=self.total_size, inputs=[1.0, b_gpu, self.r])

        wp.copy(self.z, self.r)
        wp.launch(kernel=apply_preconditioner_kernel, dim=self.total_size, inputs=[self.z, M_local_gpu])

        self.norm_buf.zero_()
        wp.launch(kernel=dot_product_kernel, dim=1024, inputs=[self.z, self.z, self.norm_buf])


        wp.copy(self.V[0], self.z)
        wp.launch(kernel=normalize_from_dot_kernel, dim=self.total_size, inputs=[self.V[0], self.norm_buf])

        self.H_gpu.zero_()
        self.cs_gpu.zero_()
        self.sn_gpu.zero_()
        self.s_gpu.zero_()
        wp.launch(write_sqrt_scalar_to_vec_kernel, dim=1, inputs=[self.norm_buf, self.s_gpu, wp.int32(0)])
        for k in range(self.max_iter):
            self.w.zero_()
            self.h_col_gpu.zero_()

            wp.launch(
                kernel=matvec_viscosity_fused_kernel,
                dim=self.Ne,
                inputs=[
                    self.V[k], self.w, U_base_gpu, M_local_gpu, Dx_phys_gpu, Dy_phys_gpu, D1d_phys_x_gpu, D1d_phys_y_gpu,
                    self.nx, self.ny, self.Ne, B_field, vg2, dt_gamma, hx, hy,
                    slip_length, qpc_f_wp, qpc_c_wp, qpc_w_wp, f_nodes_wp, norm_v_wp, w1d_wp
                ]
            )

            wp.launch(kernel=apply_preconditioner_kernel, dim=self.total_size, inputs=[self.w, M_local_gpu])

            for j in range(k + 1):
                self.dot_buf.zero_()
                wp.launch(kernel=dot_product_kernel, dim=1024, inputs=[self.w, self.V[j], self.dot_buf])

                wp.launch(
                    kernel=write_scalar_to_vec_kernel,
                    dim=1,
                    inputs=[self.dot_buf, self.h_col_gpu, wp.int32(j)]
                )

                wp.launch(
                    kernel=axpy_from_scalar_buf_kernel,
                    dim=self.total_size,
                    inputs=[self.dot_buf, self.V[j], self.w]
                )

            self.norm_buf.zero_()
            wp.launch(kernel=dot_product_kernel, dim=1024, inputs=[self.w, self.w, self.norm_buf])

            wp.launch(
                kernel=write_sqrt_scalar_to_vec_kernel,
                dim=1,
                inputs=[self.norm_buf, self.h_col_gpu, wp.int32(k + 1)]
            )

            wp.launch(kernel=copy_h_col_kernel, dim=k + 2, inputs=[self.h_col_gpu, self.H_gpu, wp.int32(k), wp.int32(self.max_iter)])
            wp.launch(kernel=apply_givens_kernel, dim=1, inputs=[self.H_gpu, self.cs_gpu, self.sn_gpu, self.s_gpu, wp.int32(k), wp.int32(self.max_iter)])
            
            wp.launch(kernel=normalize_from_dot_kernel, dim=self.total_size, inputs=[self.w, self.norm_buf])
            wp.copy(self.V[k + 1], self.w)
        wp.launch(kernel=backward_substitution_kernel, dim=1, inputs=[self.H_gpu, self.s_gpu, self.y_gpu, wp.int32(self.max_iter), wp.int32(self.max_iter)])
        for k in range(self.max_iter):
            wp.launch(kernel=axpy_from_y_vec_kernel, dim=self.total_size, inputs=[self.y_gpu, self.V[k], U_guess, wp.int32(k)])
            
        return U_guess, self.max_iter
# ============================================================================
# 1. BASIS & REFERENCE ELEMENT (p=4 GLL)
# ============================================================================
nodes1d = np.array([-1.0, -0.6546536707079771, 0.0, 0.6546536707079771, 1.0], dtype=np.float32)
weights1d = np.array([0.1, 0.5444444444444444, 0.7111111111111111, 0.5444444444444444, 0.1], dtype=np.float32)
D1d = np.array([[-5., 6.75650249, -2.66666667, 1.41016418, -0.5],[-1.24099025, 0., 1.74574312, -0.76376262, 0.25900975],[0.375, -1.33658458, 0., 1.33658458, -0.375],[-0.25900975, 0.76376262, -1.74574312, 0., 1.24099025],[0.5, -1.41016418, 2.66666667, -6.75650249, 5.]], dtype=np.float32)

W2d = np.kron(weights1d, weights1d)
Dx = np.kron(np.eye(Np), D1d)
Dy = np.kron(D1d, np.eye(Np))

face_nodes_arr = np.array([[0, 1, 2, 3, 4], [4, 9, 14, 19, 24], [20, 21, 22, 23, 24], [0, 5, 10, 15, 20]] , dtype=np.int32)
face_neighbor_map_arr = np.array([2, 3, 0, 1], dtype=np.int32)
normals_vals = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0],[-1.0, 0.0]], dtype=np.float32)

# ============================================================================
# 3. POISSON & ELECTROSTATICS
# ============================================================================
def preassemble_poisson_qpc(nx, ny, hx, hy, QPCs, Dx_phys, Dy_phys, M_local):
    Ne = nx * ny
    N = Ne * N_per_elem
    W_mat = np.diag(M_local)
    K_vol = Dx_phys.T @ W_mat @ Dx_phys + Dy_phys.T @ W_mat @ Dy_phys
    rows, cols, data = [], [],[]
    rhs_phi_static = np.zeros(N)
    penalty = 40.0 * (p_degree*p_degree) / min(hx, hy)

    for e in range(Ne):
        for i in range(N_per_elem):
            for j in range(N_per_elem):
                if abs(K_vol[i, j]) > 1e-6:
                    rows.append(e * N_per_elem + i)
                    cols.append(e * N_per_elem + j)
                    data.append(K_vol[i, j])
        ex, ey = e % nx, e // nx
        for face in range(4):
            boundary = (face == 0 and ey == 0) or (face == 2 and ey == ny - 1) or (face == 3 and ex == 0) or (face == 1 and ex == nx - 1)
            is_active, qpc_pot = False, 0.0
            if boundary:
                for q in QPCs:
                    if q['active'] and q['face'] == face:
                        e_min, e_max = (ex / nx, (ex + 1) / nx) if (face == 0 or face == 2) else (ey / ny, (ey + 1) / ny)
                        if e_max > q['center'] - q['width'] / 2 and e_min < q['center'] + q['width'] / 2:
                            is_active, qpc_pot = True, q['potential']
                            break
            if boundary and not is_active: 
                continue
            e2 = e if boundary else (e - nx if face == 0 else e + 1 if face == 1 else e + nx if face == 2 else e - 1)
            nx_f, ny_f = normals_vals[face]
            nodes1 = face_nodes_arr[face]
            nodes2 = face_nodes_arr[face_neighbor_map_arr[face]]
            ds_w = weights1d * (hx / 2.0 if (face == 0 or face == 2) else hy / 2.0)
            for q in range(Np):
                idx1, wq = nodes1[q], ds_w[q]
                if boundary:
                    phi_d = qpc_pot
                    for i in range(N_per_elem):
                        v1 = 1.0 if i == idx1 else 0.0
                        gn = Dx_phys[idx1, i] * nx_f + Dy_phys[idx1, i] * ny_f
                        rhs_phi_static[e * N_per_elem + i] += (-gn * phi_d + penalty * phi_d * v1) * wq
                        for j in range(N_per_elem):
                            u1 = 1.0 if j == idx1 else 0.0
                            gu = Dx_phys[idx1, j] * nx_f + Dy_phys[idx1, j] * ny_f
                            val = (-gu * v1 - gn * u1 + penalty * u1 * v1) * wq
                            if abs(val) > 1e-6: 
                                rows.append(e * N_per_elem + i)
                                cols.append(e * N_per_elem + j)
                                data.append(val)
                else:
                    idx2 = nodes2[q]
                    for i in range(N_per_elem):
                        v1 = 1.0 if i == idx1 else 0.0
                        gn_v1 = Dx_phys[idx1, i] * nx_f + Dy_phys[idx1, i] * ny_f
                        for j in range(N_per_elem):
                            u1 = 1.0 if j == idx1 else 0.0
                            u2 = 1.0 if j == idx2 else 0.0
                            gn_u1 = Dx_phys[idx1, j] * nx_f + Dy_phys[idx1, j] * ny_f
                            gn_u2 = Dx_phys[idx2, j] * nx_f + Dy_phys[idx2, j] * ny_f
                            b11 = (-0.5 * gn_u1 * v1 - 0.5 * gn_v1 * u1 + penalty * u1 * v1) * wq
                            b12 = (-0.5 * gn_u2 * v1 + 0.5 * gn_v1 * u2 - penalty * u2 * v1) * wq
                            if abs(b11) > 1e-6: 
                                rows.append(e * N_per_elem + i)
                                cols.append(e * N_per_elem + j)
                                data.append(b11)
                            if abs(b12) > 1e-6: 
                                rows.append(e * N_per_elem + i)
                                cols.append(e2 * N_per_elem + j)
                                data.append(b12)
    return sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr(), rhs_phi_static

# ============================================================================
# 4. MAIN INTEGRATION
# ============================================================================
def main():
    nx_grid, ny_grid = 40, 10
    Lx, Ly = 4.0, 1.0
    hx, hy = Lx / nx_grid, Ly / ny_grid
    Ne = nx_grid * ny_grid
    N = Ne * N_per_elem
    vg2 = 1.0
    e_charge = 1.0
    c_light = 1.0
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=float, default=0.5)
    parser.add_argument('--L', type=float, default=0.05)
    parser.add_argument('--V', type=float, default=0.5)
    args = parser.parse_args()
    B_field = args.B
    slip_length = args.L
    V_bias = args.V

    n0 = 1.0
    n_src, n_drn = n0, n0
    nE_src, nE_drn = np.abs(n_src)**1.5, np.abs(n_drn)**1.5
    dt = 0.002
    n_steps = 500
    gamma = 1.0 - 1.0 / np.sqrt(2.0)
    x0, w_phys = 1.0, hx
    
    qpc_configs =[
        {'face': 0, 'center': 1.0 / Lx, 'width': w_phys / Lx, 'active': True, 'type': 'source', 'potential': +V_bias / 2.0},
        {'face': 0, 'center': 3.0 / Lx, 'width': w_phys / Lx, 'active': True, 'type': 'drain', 'potential': -V_bias / 2.0},
    ]
    qpc_faces = np.array([q['face'] for q in qpc_configs], dtype=np.int32)
    qpc_centers = np.array([q['center'] for q in qpc_configs], dtype=np.float32)
    qpc_widths = np.array([q['width'] for q in qpc_configs], dtype=np.float32)
    qpc_types = np.array([0 if q['type'] == 'source' else 1 for q in qpc_configs], dtype=np.int32)

    M_local = W2d * (hx * hy / 4.0)
    Dx_phys, Dy_phys = Dx * (2.0 / hx), Dy * (2.0 / hy)

    print("Pre-assembling Poisson...")
    K_phi, rhs_phi_static_cpu = preassemble_poisson_qpc(nx_grid, ny_grid, hx, hy, qpc_configs, Dx_phys, Dy_phys, M_local)
    K_phi_csr = K_phi.tocsr()
    
    # Move CSR structure to GPU
    K_offsets = wp.array(K_phi_csr.indptr, dtype=wp.int32)
    K_cols = wp.array(K_phi_csr.indices, dtype=wp.int32)
    K_vals = wp.array(K_phi_csr.data, dtype=wp.float32)
    rhs_phi_static = wp.array(rhs_phi_static_cpu, dtype=wp.float32)
    
    phi_gpu = wp.zeros(N, dtype=wp.float32)
    r_cg = wp.zeros(N, dtype=wp.float32)
    p_cg = wp.zeros(N, dtype=wp.float32)
    Ap_cg = wp.zeros(N, dtype=wp.float32)
    cg_scalars = wp.zeros(4, dtype=wp.float32) # [rsold, rsnew, pAp, alpha]

    @wp.kernel
    def csr_spmv_kernel(offsets: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), vals: wp.array(dtype=wp.float32), x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32)):
        row = wp.tid()
        sum_val = wp.float32(0.0)
        for i in range(offsets[row], offsets[row + 1]):
            sum_val += vals[i] * x[cols[i]]
        y[row] = sum_val

    @wp.kernel
    def compute_rhs_rho_kernel(n_curr: wp.array(dtype=wp.float32), rhs_static: wp.array(dtype=wp.float32), M_loc: wp.array(dtype=wp.float32), r: wp.array(dtype=wp.float32), n0: wp.float32, e_chg: wp.float32):
        tid = wp.tid()
        r[tid] = e_chg * (n_curr[tid] - n0) * M_loc[tid % 25] + rhs_static[tid]

    @wp.kernel
    def zero_scalar_kernel(out: wp.array(dtype=wp.float32), idx: wp.int32):
        if wp.tid() == 0:
            out[idx] = wp.float32(0.0)

    @wp.kernel
    def dot_product_indexed_kernel(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32), idx: wp.int32):
        tid = wp.tid()
        sum_val = wp.float32(0.0)
        while tid < a.shape[0]:
            sum_val += a[tid] * b[tid]
            tid += 1024
        wp.atomic_add(out, idx, sum_val)

    @wp.kernel
    def cg_calc_alpha(scalars: wp.array(dtype=wp.float32)):
        if wp.tid() == 0:
            scalars[3] = scalars[0] / wp.max(scalars[2], wp.float32(1e-30))

    @wp.kernel
    def cg_update_x_r(phi: wp.array(dtype=wp.float32), r: wp.array(dtype=wp.float32), p: wp.array(dtype=wp.float32), Ap: wp.array(dtype=wp.float32), scalars: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        if tid < phi.shape[0]:
            alpha = scalars[3]
            phi[tid] = phi[tid] + alpha * p[tid]
            r[tid] = r[tid] - alpha * Ap[tid]

    @wp.kernel
    def cg_calc_beta(scalars: wp.array(dtype=wp.float32)):
        if wp.tid() == 0:
            scalars[3] = scalars[1] / wp.max(scalars[0], wp.float32(1e-30))
            scalars[0] = scalars[1]

    @wp.kernel
    def cg_update_p(p: wp.array(dtype=wp.float32), r: wp.array(dtype=wp.float32), scalars: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        if tid < p.shape[0]:
            beta = scalars[3]
            p[tid] = r[tid] + beta * p[tid]
    @wp.kernel
    def compute_E_field_kernel(phi: wp.array(dtype=wp.float32), Ex: wp.array(dtype=wp.float32), Ey: wp.array(dtype=wp.float32), Dx: wp.array(dtype=wp.float32), Dy: wp.array(dtype=wp.float32)):
        e = wp.tid()
        for i in range(25):
            ex_val, ey_val = wp.float32(0.0), wp.float32(0.0)
            for j in range(25):
                ex_val -= phi[e * 25 + j] * Dx[i * 25 + j] # Transpose applied via indexing
                ey_val -= phi[e * 25 + j] * Dy[i * 25 + j]
            Ex[e * 25 + i] = ex_val
            Ey[e * 25 + i] = ey_val

    def solve_electrostatics_gpu(U_gpu_ref):
        wp.launch(compute_rhs_rho_kernel, dim=N, inputs=[U_gpu_ref, rhs_phi_static, M_local_gpu, r_cg, wp.float32(n0), wp.float32(e_charge)])
        
        wp.launch(csr_spmv_kernel, dim=N, inputs=[K_offsets, K_cols, K_vals, phi_gpu, Ap_cg])
        wp.launch(axpy_kernel, dim=N, inputs=[-1.0, Ap_cg, r_cg])
        wp.copy(p_cg, r_cg)
        
        wp.launch(zero_scalar_kernel, dim=1, inputs=[cg_scalars, wp.int32(0)])
        wp.launch(dot_product_indexed_kernel, dim=1024, inputs=[r_cg, r_cg, cg_scalars, wp.int32(0)])
        
        for _ in range(50): 
            wp.launch(csr_spmv_kernel, dim=N, inputs=[K_offsets, K_cols, K_vals, p_cg, Ap_cg])
            
            wp.launch(zero_scalar_kernel, dim=1, inputs=[cg_scalars, wp.int32(2)])
            wp.launch(dot_product_indexed_kernel, dim=1024, inputs=[p_cg, Ap_cg, cg_scalars, wp.int32(2)])
            
            wp.launch(cg_calc_alpha, dim=1, inputs=[cg_scalars])
            wp.launch(cg_update_x_r, dim=N, inputs=[phi_gpu, r_cg, p_cg, Ap_cg, cg_scalars])
            
            wp.launch(zero_scalar_kernel, dim=1, inputs=[cg_scalars, wp.int32(1)])
            wp.launch(dot_product_indexed_kernel, dim=1024, inputs=[r_cg, r_cg, cg_scalars, wp.int32(1)])
            
            wp.launch(cg_calc_beta, dim=1, inputs=[cg_scalars])
            wp.launch(cg_update_p, dim=N, inputs=[p_cg, r_cg, cg_scalars])
        
        wp.launch(compute_E_field_kernel, dim=Ne, inputs=[phi_gpu, Ex_gpu, Ey_gpu, Dx_phys_gpu, Dy_phys_gpu])
    
    U_n = np.zeros(4 * N, dtype=np.float32)
    U_n[0:N] = n0
    U_n[N:2*N] = nE_src 

    def fmt_tag(x): 
        return f"{x:.3f}".replace("-", "m").replace(".", "p")
        
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    run_tag = (f"nx{nx_grid}_ny{ny_grid}_B{fmt_tag(B_field)}_slip{fmt_tag(slip_length)}"
               f"_V{fmt_tag(V_bias)}_dt{fmt_tag(dt)}_steps{n_steps}_{run_stamp}")
    output_dir = f"sim_data_{run_tag}"
    snapshot_dir = os.path.join(output_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    diagnostics_path = os.path.join(output_dir, "diagnostics.csv")
    with open(diagnostics_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "time", "n_min", "n_max", "phi_min", "phi_max", "jmag_max", "omega_max", "charge_excess", "momentum_max", "gmres_stage1", "gmres_stage2"])
        

    qpc_face = np.array([q['face'] for q in qpc_configs], dtype=np.int32)
    qpc_center = np.array([q['center'] for q in qpc_configs], dtype=np.float32)
    qpc_width = np.array([q['width'] for q in qpc_configs], dtype=np.float32)
    qpc_potential = np.array([q['potential'] for q in qpc_configs], dtype=np.float32)
    qpc_kind = np.array([1 if q['type'] == 'source' else -1 if q['type'] == 'drain' else 0 
                         for q in qpc_configs], dtype=np.int32)

    np.savez_compressed(
        os.path.join(output_dir, "meta.npz"),
        run_tag=run_tag,
        run_stamp=run_stamp,
        nx=np.int32(nx_grid),
        ny=np.int32(ny_grid),
        Lx=np.float32(Lx),
        Ly=np.float32(Ly),
        hx=np.float32(hx),
        hy=np.float32(hy),
        dt=np.float32(dt),
        n_steps=np.int32(n_steps),
        vg2=np.float32(vg2),
        e_charge=np.float32(e_charge),
        c_light=np.float32(c_light),
        B_field=np.float32(B_field),
        V_bias=np.float32(V_bias),
        n0=np.float32(n0),
        slip_length=np.float32(slip_length),
        p_degree=np.int32(p_degree),
        Np=np.int32(Np),
        N_per_elem=np.int32(N_per_elem),
        nodes1d=nodes1d,
        weights1d=weights1d,
        M_local=M_local,
        Dx_phys=Dx_phys,
        Dy_phys=Dy_phys,
        qpc_face=qpc_face,
        qpc_center=qpc_center,
        qpc_width=qpc_width,
        qpc_potential=qpc_potential,
        qpc_kind=qpc_kind,
    )

    def dump_snapshot(step, U_current, phi_current, Ex_current, Ey_current, gmres1=np.nan, gmres2=np.nan):
        n_nodes = U_current[0:N].reshape(Ne, N_per_elem)
        nE_nodes = U_current[N:2*N].reshape(Ne, N_per_elem)
        gx_nodes = U_current[2*N:3*N].reshape(Ne, N_per_elem)
        gy_nodes = U_current[3*N:4*N].reshape(Ne, N_per_elem)
    
        u_s = np.zeros((Ne, N_per_elem))
        v_s = np.zeros((Ne, N_per_elem))
        P_s = np.zeros((Ne, N_per_elem))
        W_s = np.zeros((Ne, N_per_elem))
    
        for e in range(Ne):
            for i in range(N_per_elem):
                n_val = n_nodes[e, i]
                nE_val = nE_nodes[e, i]
                gx_val = gx_nodes[e, i]
                gy_val = gy_nodes[e, i]
                
                g_norm = np.sqrt(gx_val*gx_val + gy_val*gy_val)
                nE_safe = max(nE_val, 1e-6)
                if g_norm > 1e-6:
                    disc = 9.0 * nE_safe * nE_safe - 8.0 * g_norm * g_norm * vg2
                    u_norm = (3.0 * nE_safe - np.sqrt(max(disc, 0.0))) / (2.0 * g_norm)
                else:
                    u_norm = (2.0 * g_norm * vg2) / (3.0 * nE_safe)
                    
                u_norm = min(u_norm, 0.999 * np.sqrt(vg2))
                v2 = (u_norm * u_norm) / vg2
                w_val = 3.0 * nE_safe / (2.0 + v2 + 1e-6)
                p_val = nE_safe * (1.0 - v2) / (2.0 + v2 + 1e-6)
                
                if g_norm > 1e-6:
                    ux = (gx_val / g_norm) * u_norm
                    uy = (gy_val / g_norm) * u_norm
                else:
                    ux, uy = 0.0, 0.0
                    
                u_s[e, i], v_s[e, i], P_s[e, i], W_s[e, i] = ux, uy, p_val, w_val
    
        jx_nodes = n_nodes * u_s
        jy_nodes = n_nodes * v_s
        jmag_nodes = np.sqrt(jx_nodes * jx_nodes + jy_nodes * jy_nodes)
        omega_nodes = (v_s @ Dx_phys.T) - (u_s @ Dy_phys.T)
    
        np.savez_compressed(
            os.path.join(snapshot_dir, f"step_{step:06d}.npz"),
            step=np.int32(step),
            time=np.float32(step * dt),
            n=n_nodes,
            nE=nE_nodes,
            gx=gx_nodes,
            gy=gy_nodes,
            u=u_s,
            v=v_s,
            jx=jx_nodes,
            jy=jy_nodes,
            jmag=jmag_nodes,
            omega=omega_nodes,
            phi=phi_current.reshape(Ne, N_per_elem),
            Ex=Ex_current.reshape(Ne, N_per_elem),
            Ey=Ey_current.reshape(Ne, N_per_elem),
        )
    
        charge_excess = np.sum((n_nodes - n0) * M_local[None, :])
        momentum_max = np.max(np.sqrt(gx_nodes * gx_nodes + gy_nodes * gy_nodes))
    
        with open(diagnostics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                step * dt,
                np.min(n_nodes),
                np.max(n_nodes),
                np.min(phi_current),
                np.max(phi_current),
                np.max(jmag_nodes),
                np.max(np.abs(omega_nodes)),
                charge_excess,
                momentum_max,
                gmres1,
                gmres2,
            ])

    
    print("Starting GPU optimization mapping...")
    start_time = time.time()
    warp_solver = WarpGMRES(nx_grid, ny_grid)

    M_local_gpu = wp.array(M_local.astype(np.float32), dtype=wp.float32)
    Dx_phys_gpu = wp.array(Dx_phys.flatten().astype(np.float32), dtype=wp.float32)
    Dy_phys_gpu = wp.array(Dy_phys.flatten().astype(np.float32), dtype=wp.float32)
    D1d_phys_x_gpu = wp.array((D1d * (2.0 / hx)).flatten().astype(np.float32), dtype=wp.float32)
    D1d_phys_y_gpu = wp.array((D1d * (2.0 / hy)).flatten().astype(np.float32), dtype=wp.float32)
    qpc_f_gpu = wp.array(qpc_faces.astype(np.int32), dtype=wp.int32)
    qpc_c_gpu = wp.array(qpc_centers.astype(np.float32), dtype=wp.float32)
    qpc_w_gpu = wp.array(qpc_widths.astype(np.float32), dtype=wp.float32)
    qpc_t_gpu = wp.array(qpc_types.astype(np.int32), dtype=wp.int32)
    f_nodes_gpu = wp.array(face_nodes_arr.astype(np.int32), dtype=wp.int32, ndim=2)
    norm_v_gpu = wp.array(normals_vals.astype(np.float32), dtype=wp.float32, ndim=2)
    w1d_gpu = wp.array(weights1d.astype(np.float32), dtype=wp.float32)

    U_gpu = wp.array(U_n.astype(np.float32), dtype=wp.float32)
    U_1_gpu = wp.zeros_like(U_gpu)
    rhs_n_gpu = wp.zeros_like(U_gpu)
    rhs_1_gpu = wp.zeros_like(U_gpu)
    b_stage_gpu = wp.zeros_like(U_gpu)
    
    Ex_gpu = wp.zeros(N, dtype=wp.float32)
    Ey_gpu = wp.zeros(N, dtype=wp.float32)

    # ------------------------------------------------------------------
    # Add these once, after Ex_gpu / Ey_gpu allocation
    # ------------------------------------------------------------------
    # CPU buffers for diagnostic extraction (moved up so we can dump step 0)
    Ufinal_host = wp.zeros(4 * N, dtype=wp.float32, device="cpu")
    phi_host = wp.zeros(N, dtype=wp.float32, device="cpu")
    Ex_host = wp.zeros(N, dtype=wp.float32, device="cpu")
    Ey_host = wp.zeros(N, dtype=wp.float32, device="cpu")

    print("Pre-converging initial electrostatics...")
    phi_gpu.zero_()
    # Run the solver 20 times (1000 total iterations) to perfectly smooth the initial field
    for _ in range(20): 
        solve_electrostatics_gpu(U_gpu)

    # Optional: Dump the fully converged 0th state
    wp.copy(Ufinal_host, U_gpu)
    wp.copy(phi_host, phi_gpu)
    wp.copy(Ex_host, Ex_gpu)
    wp.copy(Ey_host, Ey_gpu)
    wp.synchronize()
    dump_snapshot(0, Ufinal_host.numpy(), phi_host.numpy(), Ex_host.numpy(), Ey_host.numpy())
    print("Capturing CUDA Graph...")
    wp.capture_begin()

    # STAGE 1
    solve_electrostatics_gpu(U_gpu)
    rhs_n_gpu.zero_()
    wp.launch(compute_explicit_euler_flux_kernel, dim=Ne, inputs=[
        U_gpu, rhs_n_gpu, M_local_gpu, Dx_phys_gpu, Dy_phys_gpu,
        nx_grid, ny_grid, Ne, wp.float32(vg2),
        wp.float32(n_src), wp.float32(nE_src), wp.float32(n_drn), wp.float32(nE_drn),
        qpc_f_gpu, qpc_c_gpu, qpc_w_gpu, qpc_t_gpu, f_nodes_gpu, norm_v_gpu, w1d_gpu, wp.float32(hx), wp.float32(hy)])
    wp.launch(compute_electromagnetic_source_kernel, dim=Ne, inputs=[
        U_gpu, rhs_n_gpu, M_local_gpu, Ex_gpu, Ey_gpu, Ne, wp.float32(B_field), wp.float32(vg2), wp.float32(e_charge), wp.float32(c_light)])

    wp.launch(compute_b_stage1_kernel, dim=4*N, inputs=[
        U_gpu, rhs_n_gpu, M_local_gpu, wp.float32(dt * gamma), b_stage_gpu])
    wp.launch(clamp_b_state_kernel, dim=Ne, inputs=[b_stage_gpu, Ne, wp.float32(vg2)])

    wp.copy(U_1_gpu, U_gpu) 
    warp_solver.solve(
        U_1_gpu, b_stage_gpu, wp.float32(dt * gamma), U_gpu, M_local_gpu, Dx_phys_gpu, Dy_phys_gpu, D1d_phys_x_gpu, D1d_phys_y_gpu, 
        wp.float32(B_field), wp.float32(vg2), wp.float32(hx), wp.float32(hy), wp.float32(slip_length), 
        qpc_f_gpu, qpc_c_gpu, qpc_w_gpu, f_nodes_gpu, norm_v_gpu, w1d_gpu
    ) 

    # STAGE 2
    solve_electrostatics_gpu(U_1_gpu)
    rhs_1_gpu.zero_()
    wp.launch(compute_explicit_euler_flux_kernel, dim=Ne, inputs=[
        U_1_gpu, rhs_1_gpu, M_local_gpu, Dx_phys_gpu, Dy_phys_gpu,
        nx_grid, ny_grid, Ne, wp.float32(vg2),
        wp.float32(n_src), wp.float32(nE_src), wp.float32(n_drn), wp.float32(nE_drn),
        qpc_f_gpu, qpc_c_gpu, qpc_w_gpu, qpc_t_gpu, f_nodes_gpu, norm_v_gpu, w1d_gpu, wp.float32(hx), wp.float32(hy)])
    wp.launch(compute_electromagnetic_source_kernel, dim=Ne, inputs=[
        U_1_gpu, rhs_1_gpu, M_local_gpu, Ex_gpu, Ey_gpu, Ne, wp.float32(B_field), wp.float32(vg2), wp.float32(e_charge), wp.float32(c_light)])
    
    c_n = (2.0 * gamma - 1.0) / gamma
    c_1 = (1.0 - gamma) / gamma
    dt_gamma2 = dt * gamma
    wp.launch(compute_b_stage2_kernel, dim=4*N, inputs=[
        U_gpu, U_1_gpu, rhs_1_gpu, M_local_gpu, wp.float32(dt_gamma2), wp.float32(c_n), wp.float32(c_1), b_stage_gpu])
    wp.launch(clamp_b_state_kernel, dim=Ne, inputs=[b_stage_gpu, Ne, wp.float32(vg2)])

    warp_solver.solve(
        U_1_gpu, b_stage_gpu, wp.float32(dt * gamma), U_1_gpu, M_local_gpu, Dx_phys_gpu, Dy_phys_gpu, D1d_phys_x_gpu, D1d_phys_y_gpu, 
        wp.float32(B_field), wp.float32(vg2), wp.float32(hx), wp.float32(hy), wp.float32(slip_length), 
        qpc_f_gpu, qpc_c_gpu, qpc_w_gpu, f_nodes_gpu, norm_v_gpu, w1d_gpu
    )
    wp.copy(U_gpu, U_1_gpu)
    graph = wp.capture_end()

    # CPU buffers for diagnostic extraction
    Ufinal_host = wp.zeros(4 * N, dtype=wp.float32, device="cpu")
    phi_host = wp.zeros(N, dtype=wp.float32, device="cpu")
    Ex_host = wp.zeros(N, dtype=wp.float32, device="cpu")
    Ey_host = wp.zeros(N, dtype=wp.float32, device="cpu")

    print("Starting implicit integration...")
    for step in range(n_steps):
        wp.capture_launch(graph)
        
        if step % 10 == 0 or step == n_steps - 1:
            wp.copy(Ufinal_host, U_gpu)
            wp.copy(phi_host, phi_gpu)
            wp.copy(Ex_host, Ex_gpu)
            wp.copy(Ey_host, Ey_gpu)
            wp.synchronize()
            
            U_final_cpu = Ufinal_host.numpy()
            phi_final = phi_host.numpy()
            Ex_final = Ex_host.numpy()
            Ey_final = Ey_host.numpy()
            
            dump_snapshot(step + 1, U_final_cpu, phi_final, Ex_final, Ey_final, warp_solver.max_iter, warp_solver.max_iter)
            print(f"Step {step:03d} | Max Momentum (g): {np.max(np.abs(U_final_cpu[2*N:4*N])):.4e}")
            
    print(f"Simulation finished in {time.time() - start_time:.2f} seconds.")
if __name__ == '__main__':
    main()
