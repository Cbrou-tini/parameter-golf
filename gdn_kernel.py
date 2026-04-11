import mlx.core as mx

# ==============================================================================
# Metal kernel sources
# ==============================================================================

INTRA_SOURCE = """
    uint bh    = thread_position_in_grid.x;
    uint chunk = thread_position_in_grid.y;
    uint d     = thread_position_in_grid.z;

    uint t_start = chunk * C;

    float S_row[D];
    for (uint dd = 0; dd < D; dd++)
        S_row[dd] = S_incoming[((bh * num_chunks + chunk) * D + d) * D + dd];

    for (uint i = 0; i < C; i++) {
        uint t = t_start + i;
        float a_i = alpha[bh * T + t];
        float b_i = beta[bh * T + t];
        float kv = 0.0f;
        for (uint dd = 0; dd < D; dd++)
            kv += S_row[dd] * k_s[(bh * T + t) * D + dd];
        float delta_d = v_s[(bh * T + t) * D + d] - kv;
        for (uint dd = 0; dd < D; dd++)
            S_row[dd] = a_i * S_row[dd] + b_i * delta_d * k_s[(bh * T + t) * D + dd];

        float val = 0.0f;
        for (uint dd = 0; dd < D; dd++)
            val += S_row[dd] * q_s[(bh * T + t) * D + dd];
        intra_out[(bh * T + t) * D + d] = val;
    }

    for (uint dd = 0; dd < D; dd++)
        chunk_states[((bh * num_chunks + chunk) * D + d) * D + dd] = S_row[dd];
"""

INTRA_SAVE_SOURCE = """
    uint bh    = thread_position_in_grid.x;
    uint chunk = thread_position_in_grid.y;
    uint d     = thread_position_in_grid.z;

    uint t_start = chunk * C;

    float S_row[D];
    for (uint dd = 0; dd < D; dd++)
        S_row[dd] = S_incoming[((bh * num_chunks + chunk) * D + d) * D + dd];

    for (uint i = 0; i < C; i++) {
        uint t = t_start + i;
        float a_i = alpha[bh * T + t];
        float b_i = beta[bh * T + t];
        float kv = 0.0f;
        for (uint dd = 0; dd < D; dd++)
            kv += S_row[dd] * k_s[(bh * T + t) * D + dd];
        float delta_d = v_s[(bh * T + t) * D + d] - kv;

        delta_hist[(bh * T + t) * D + d] = delta_d;

        for (uint dd = 0; dd < D; dd++)
            S_row[dd] = a_i * S_row[dd] + b_i * delta_d * k_s[(bh * T + t) * D + dd];

        for (uint dd = 0; dd < D; dd++)
            S_hist[(bh * T + t) * D * D + d * D + dd] = S_row[dd];

        float val = 0.0f;
        for (uint dd = 0; dd < D; dd++)
            val += S_row[dd] * q_s[(bh * T + t) * D + dd];
        intra_out[(bh * T + t) * D + d] = val;
    }

    for (uint dd = 0; dd < D; dd++)
        chunk_states[((bh * num_chunks + chunk) * D + d) * D + dd] = S_row[dd];
"""

INTER_FWD_SCAN_SOURCE = """
    uint bh    = thread_position_in_grid.x;
    uint d_row = thread_position_in_grid.y;

    float S_row[D];
    for (uint dc = 0; dc < D; dc++) S_row[dc] = 0.0f;

    for (uint chunk = 0; chunk < num_chunks; chunk++) {
        for (uint dc = 0; dc < D; dc++)
            S_incoming[((bh * num_chunks + chunk) * D + d_row) * D + dc] = S_row[dc];

        float a_chunk = 1.0f;
        uint t_start = chunk * C;
        for (uint i = 0; i < C; i++)
            a_chunk *= alpha[bh * T + t_start + i];

        for (uint dc = 0; dc < D; dc++)
            S_row[dc] = a_chunk * S_row[dc] + chunk_states[((bh * num_chunks + chunk) * D + d_row) * D + dc];
    }
"""

INTRA_BWD_SOURCE = """
    uint bh    = thread_position_in_grid.x;
    uint chunk = thread_position_in_grid.y;
    uint d     = thread_position_in_grid.z;
    uint t_start = chunk * C;

    float da_local[C];
    float db_local[C];
    for (uint i = 0; i < C; i++) { da_local[i] = 0.0f; db_local[i] = 0.0f; }

    float dS_row[D];
    for (uint dc = 0; dc < D; dc++)
        dS_row[dc] = dchunk_states[((bh * num_chunks + chunk) * D + d) * D + dc];

    for (int i = (int)C - 1; i >= 0; i--) {
        uint t = t_start + (uint)i;
        float a_i     = alpha[bh * T + t];
        float b_i     = beta[bh * T + t];
        float delta_d = delta_hist[(bh * T + t) * D + d];

        float dout_d = dout[(bh * T + t) * D + d];
        for (uint dc = 0; dc < D; dc++)
            dS_row[dc] += dout_d * q_s[(bh * T + t) * D + dc];

        float dq_val = 0.0f;
        for (uint dc = 0; dc < D; dc++)
            dq_val += S_hist[(bh * T + t) * D * D + dc * D + d]
                    * dout[(bh * T + t) * D + dc];
        atomic_fetch_add_explicit((device atomic<float>*)&dq_s[(bh * T + t) * D + d],
                                   dq_val, memory_order_relaxed);

        float ddelta = 0.0f;
        for (uint dc = 0; dc < D; dc++)
            ddelta += b_i * dS_row[dc] * k_s[(bh * T + t) * D + dc];

        atomic_fetch_add_explicit((device atomic<float>*)&dv_s[(bh * T + t) * D + d],
                                   ddelta, memory_order_relaxed);

        float da_c = 0.0f, db_c = 0.0f;
        for (uint dc = 0; dc < D; dc++) {
            float S_prev = (i > 0)
                ? S_hist[(bh * T + (t - 1)) * D * D + d * D + dc]
                : S_incoming[((bh * num_chunks + chunk) * D + d) * D + dc];
            da_c += dS_row[dc] * S_prev;
            db_c += dS_row[dc] * delta_d * k_s[(bh * T + t) * D + dc];
        }
        da_local[i] += da_c;
        db_local[i] += db_c;

        for (uint dc = 0; dc < D; dc++) {
            float S_prev = (i > 0)
                ? S_hist[(bh * T + (t - 1)) * D * D + d * D + dc]
                : S_incoming[((bh * num_chunks + chunk) * D + d) * D + dc];
            atomic_fetch_add_explicit((device atomic<float>*)&dk_s[(bh * T + t) * D + dc],
                                       b_i * delta_d * dS_row[dc] - S_prev * ddelta,
                                       memory_order_relaxed);
        }

        for (uint dc = 0; dc < D; dc++)
            dS_row[dc] = a_i * dS_row[dc] - k_s[(bh * T + t) * D + dc] * ddelta;
    }

    for (uint dc = 0; dc < D; dc++)
        atomic_store_explicit((device atomic<float>*)&dS_out[((bh * num_chunks + chunk) * D + d) * D + dc],
                   dS_row[dc], memory_order_relaxed);

    for (uint i = 0; i < C; i++) {
        uint t = t_start + i;
        atomic_fetch_add_explicit((device atomic<float>*)&dalpha[bh * T + t],
                                   da_local[i], memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic<float>*)&dbeta[bh * T + t],
                                   db_local[i], memory_order_relaxed);
    }
"""

INTER_BWD_FUSED_SOURCE = """
    uint bh    = thread_position_in_grid.x;
    uint d_row = thread_position_in_grid.y;

    float dS_row[D];
    for (uint dc = 0; dc < D; dc++) dS_row[dc] = 0.0f;

    for (int chunk = (int)num_chunks - 1; chunk >= 0; chunk--) {
        uint t_start = chunk * C;

        for (uint dc = 0; dc < D; dc++)
            atomic_store_explicit((device atomic<float>*)&dchunk_states[((bh * num_chunks + chunk) * D + d_row) * D + dc],
                       dS_row[dc], memory_order_relaxed);

        float a_chunk = 1.0f;
        for (uint i = 0; i < C; i++)
            a_chunk *= alpha[bh * T + t_start + i];

        float dot_val = 0.0f;
        for (uint dc = 0; dc < D; dc++)
            dot_val += dS_row[dc]
                     * S_incoming[((bh * num_chunks + chunk) * D + d_row) * D + dc];

        for (uint i = 0; i < C; i++) {
            uint t = t_start + i;
            float da_t = (a_chunk / alpha[bh * T + t]) * dot_val;
            atomic_fetch_add_explicit((device atomic<float>*)&dalpha[bh * T + t],
                                       da_t, memory_order_relaxed);
        }

        for (uint dc = 0; dc < D; dc++)
            dS_row[dc] = a_chunk * dS_row[dc] + dS_local[((bh * num_chunks + chunk) * D + d_row) * D + dc];
    }
"""

_fwd_kernel_cache: dict = {}
_bwd_kernel_cache: dict = {}
_scan_kernel_cache: dict = {}


def _get_fwd_kernels(D):
    if D not in _fwd_kernel_cache:
        header = f"#define C 64\n#define D {D}\n"
        intra = mx.fast.metal_kernel(
            name=f"gdn_intra_chunk_D{D}",
            input_names=["k_s", "v_s", "q_s", "beta", "alpha", "S_incoming"],
            output_names=["intra_out", "chunk_states"],
            source=INTRA_SOURCE,
            header=header,
        )
        intra_save = mx.fast.metal_kernel(
            name=f"gdn_intra_save_D{D}",
            input_names=["k_s", "v_s", "q_s", "beta", "alpha", "S_incoming"],
            output_names=["intra_out", "chunk_states", "S_hist", "delta_hist"],
            source=INTRA_SAVE_SOURCE,
            header=header,
        )
        _fwd_kernel_cache[D] = (intra, intra_save)
    return _fwd_kernel_cache[D]


def _get_scan_kernel(D):
    if D not in _scan_kernel_cache:
        header = f"#define C 64\n#define D {D}\n"
        scan = mx.fast.metal_kernel(
            name=f"gdn_inter_fwd_scan_D{D}",
            input_names=["chunk_states", "alpha"],
            output_names=["S_incoming"],
            source=INTER_FWD_SCAN_SOURCE,
            header=header,
        )
        _scan_kernel_cache[D] = scan
    return _scan_kernel_cache[D]


def _get_bwd_kernels(D):
    if D not in _bwd_kernel_cache:
        header = f"#define C 64\n#define D {D}\n"
        intra_bwd = mx.fast.metal_kernel(
            name=f"gdn_intra_bwd_D{D}",
            input_names=["dout", "k_s", "v_s", "q_s", "beta", "alpha",
                         "S_hist", "delta_hist", "S_incoming", "dchunk_states"],
            output_names=["dk_s", "dv_s", "dq_s", "dalpha", "dbeta", "dS_out"],
            source=INTRA_BWD_SOURCE,
            header=header,
            atomic_outputs=True,
        )
        inter_bwd_fused = mx.fast.metal_kernel(
            name=f"gdn_inter_bwd_fused_D{D}",
            input_names=["dS_local", "alpha", "S_incoming"],
            output_names=["dchunk_states", "dalpha"],
            source=INTER_BWD_FUSED_SOURCE,
            header=header,
            atomic_outputs=True,
        )
        _bwd_kernel_cache[D] = (intra_bwd, inter_bwd_fused)
    return _bwd_kernel_cache[D]


def gdn_forward_and_save(k_s, v_s, q_s, beta, alpha, C=64):
    BH, T, D = k_s.shape
    assert T % C == 0
    num_chunks = T // C

    k_s   = mx.contiguous(k_s.astype(mx.float32))
    v_s   = mx.contiguous(v_s.astype(mx.float32))
    q_s   = mx.contiguous(q_s.astype(mx.float32))
    beta  = mx.contiguous(beta.astype(mx.float32))
    alpha = mx.contiguous(alpha.astype(mx.float32))

    _intra_kernel, _intra_save_kernel = _get_fwd_kernels(D)
    _scan_kernel = _get_scan_kernel(D)

    S_incoming_zero = mx.zeros((BH, num_chunks, D, D), dtype=mx.float32)
    _, chunk_states = _intra_kernel(
        inputs=[k_s, v_s, q_s, beta, alpha, S_incoming_zero],
        template=[("T", T), ("num_chunks", num_chunks)],
        grid=(BH, num_chunks, D),
        threadgroup=(1, 1, 1),
        output_shapes=[(BH, T, D), (BH, num_chunks, D, D)],
        output_dtypes=[mx.float32, mx.float32],
    )

    chunk_states = mx.contiguous(chunk_states)
    S_incoming = _scan_kernel(
        inputs=[chunk_states, alpha],
        template=[("T", T), ("num_chunks", num_chunks)],
        grid=(BH, D, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(BH, num_chunks, D, D)],
        output_dtypes=[mx.float32],
    )[0]

    S_incoming = mx.contiguous(S_incoming)

    intra_out, _, S_hist, delta_hist = _intra_save_kernel(
        inputs=[k_s, v_s, q_s, beta, alpha, S_incoming],
        template=[("T", T), ("num_chunks", num_chunks)],
        grid=(BH, num_chunks, D),
        threadgroup=(1, 1, 1),
        output_shapes=[(BH, T, D), (BH, num_chunks, D, D), (BH, T, D, D), (BH, T, D)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
    )

    return intra_out, S_hist, delta_hist, S_incoming


def gdn_chunkwise_scan(k_s, v_s, q_s, beta, alpha, C=64):
    out, _, _, _ = gdn_forward_and_save(k_s, v_s, q_s, beta, alpha, C)
    return out


def gdn_backward_metal(dout, k_s, v_s, q_s, beta, alpha, S_hist, delta_hist, S_incoming, C=64):
    BH, T, D = k_s.shape
    num_chunks = T // C

    dout       = mx.contiguous(dout.astype(mx.float32))
    k_s        = mx.contiguous(k_s.astype(mx.float32))
    v_s        = mx.contiguous(v_s.astype(mx.float32))
    q_s        = mx.contiguous(q_s.astype(mx.float32))
    beta       = mx.contiguous(beta.astype(mx.float32))
    alpha      = mx.contiguous(alpha.astype(mx.float32))
    S_hist     = mx.contiguous(S_hist.astype(mx.float32))
    delta_hist = mx.contiguous(delta_hist.astype(mx.float32))
    S_incoming = mx.contiguous(S_incoming.astype(mx.float32))

    _intra_bwd_kernel, _inter_bwd_fused_kernel = _get_bwd_kernels(D)

    dchunk_zero = mx.zeros((BH, num_chunks, D, D), dtype=mx.float32)
    _, _, _, dalpha_pass1, _, dS_local = _intra_bwd_kernel(
        inputs=[dout, k_s, v_s, q_s, beta, alpha,
                S_hist, delta_hist, S_incoming, dchunk_zero],
        template=[("T", T), ("num_chunks", num_chunks)],
        grid=(BH, num_chunks, D),
        threadgroup=(1, 1, 1),
        output_shapes=[(BH, T, D), (BH, T, D), (BH, T, D), (BH, T), (BH, T),
                       (BH, num_chunks, D, D)],
        output_dtypes=[mx.float32] * 6,
        init_value=0,
    )

    dS_local = mx.contiguous(dS_local)
    dchunk_states, dalpha_inter = _inter_bwd_fused_kernel(
        inputs=[dS_local, alpha, S_incoming],
        template=[("T", T), ("num_chunks", num_chunks)],
        grid=(BH, D, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(BH, num_chunks, D, D), (BH, T)],
        output_dtypes=[mx.float32, mx.float32],
        init_value=0,
    )

    dchunk_states = mx.contiguous(dchunk_states)
    dk, dv, dq, dalpha_pass2, dbeta, _ = _intra_bwd_kernel(
        inputs=[dout, k_s, v_s, q_s, beta, alpha,
                S_hist, delta_hist, S_incoming, dchunk_states],
        template=[("T", T), ("num_chunks", num_chunks)],
        grid=(BH, num_chunks, D),
        threadgroup=(1, 1, 1),
        output_shapes=[(BH, T, D), (BH, T, D), (BH, T, D), (BH, T), (BH, T),
                       (BH, num_chunks, D, D)],
        output_dtypes=[mx.float32] * 6,
        init_value=0,
    )

    dalpha = dalpha_pass2 + dalpha_inter

    return dk, dv, dq, dbeta, dalpha


def gdn_backward(dout, k_s, v_s, q_s, beta, alpha):
    _, S_hist, delta_hist, S_incoming = gdn_forward_and_save(k_s, v_s, q_s, beta, alpha)
    return gdn_backward_metal(dout, k_s, v_s, q_s, beta, alpha, S_hist, delta_hist, S_incoming)


@mx.custom_function
def gdn_forward_custom(k_s, v_s, q_s, beta, alpha):
    out = gdn_chunkwise_scan(k_s, v_s, q_s, beta, alpha)
    return out.astype(k_s.dtype)

@gdn_forward_custom.vjp
def gdn_vjp(primals, cotangents, output):
    k_s, v_s, q_s, beta, alpha = primals
    dout = cotangents.astype(mx.float32)
    dk, dv, dq, dbeta, dalpha = gdn_backward(
        dout, k_s, v_s, q_s, beta, alpha
    )
    return (dk.astype(k_s.dtype), dv.astype(v_s.dtype), dq.astype(q_s.dtype),
            dbeta.astype(beta.dtype), dalpha.astype(alpha.dtype))


def gdn_reference(k_s, v_s, q_s, beta, alpha):
    BH, T, D = k_s.shape
    S = mx.zeros((BH, D, D), dtype=mx.float32)
    outputs = []
    for i in range(T):
        ki = k_s[:, i, :]
        vi = v_s[:, i, :]
        qi = q_s[:, i, :]
        kv_mem = (S @ ki[:, :, None]).squeeze(-1)
        delta  = vi - kv_mem
        bi     = beta[:, i, None, None]
        ai     = alpha[:, i, None, None]
        S = ai * S + bi * (delta[:, :, None] * ki[:, None, :])
        out_i = (S @ qi[:, :, None]).squeeze(-1)
        outputs.append(out_i[:, None, :])
    return mx.concatenate(outputs, axis=1)
