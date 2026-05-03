use half::f16;
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{
        Device, CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU,
        CL_DEVICE_TYPE_GPU,
    },
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_WRITE},
    platform::get_platforms,
    program::Program,
    types::{cl_device_id, cl_int},
};
use std::fs;

const CL_BASE_ARGS: &str =
    "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-denorms-are-zero -DQ4K_B";

// Fixed problem dimensions
const BATCH_SIZE: usize = 36;
const M: usize = 256; // kSizeM
const N: usize = 32;  // kSizeN
const K: usize = 256; // kSizeK  (must equal QK_K=256)

// ── Utility ──────────────────────────────────────────────────────────────────

fn next_power_of_two(x: usize) -> usize {
    if x <= 1 {
        return 1;
    }
    1_usize << (usize::BITS - (x - 1).leading_zeros()) as usize
}

fn parse_opencl_version(s: &str) -> f32 {
    s.split_whitespace()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0)
}

fn device_type_str(t: u64) -> &'static str {
    if t == CL_DEVICE_TYPE_GPU {
        "GPU"
    } else if t == CL_DEVICE_TYPE_CPU {
        "CPU"
    } else if t == CL_DEVICE_TYPE_ACCELERATOR {
        "Accelerator"
    } else {
        "Unknown"
    }
}

// ── Data helpers ─────────────────────────────────────────────────────────────

fn generate_data(
    out: &mut [f16],
    m: usize,
    n: usize,
    batch_size: usize,
    m_ceil: usize,
    n_ceil: usize,
) {
    for batch in 0..batch_size {
        for i in 0..n_ceil {
            for j in 0..m_ceil {
                let val = if i < n && j < m {
                    (((i ^ j) as i32 + batch as i32 - 128) % 256) as f32 / 256.0
                } else {
                    0.0
                };
                out[batch * n_ceil * m_ceil + i * m_ceil + j] = f16::from_f32(val);
            }
        }
    }
}

fn reference_gemm(
    a: &[f16],
    b: &[f16],
    c: &mut [f16],
    m: usize,
    n: usize,
    k: usize,
    batch_size: usize,
) {
    let af: Vec<f32> = a.iter().map(|&v| f32::from(v)).collect();
    let bf: Vec<f32> = b.iter().map(|&v| f32::from(v)).collect();
    let mut cf = vec![0.0f32; batch_size * m * n];

    for batch in 0..batch_size {
        let oa = batch * m * k;
        let ob = batch * n * k;
        let oc = batch * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    acc += af[l * m + i + oa] * bf[l * n + j + ob];
                }
                cf[j * m + i + oc] = acc;
            }
        }
    }

    for (dst, &src) in c.iter_mut().zip(cf.iter()) {
        *dst = f16::from_f32(src);
    }
}

fn compare_ref(
    x: &[f16],
    r: &[f16],
    m: usize,
    n: usize,
    batch_size: usize,
    m_ceil: usize,
    n_ceil: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for batch in 0..batch_size {
        for j in 0..m {
            for i in 0..n {
                let rv = f32::from(r[batch * n * m + j * n + i]);
                let xv = f32::from(x[batch * n_ceil * m_ceil + j * n_ceil + i]);
                sum += (rv - xv) * (rv - xv);
            }
        }
    }
    sum / (m * n * batch_size) as f32
}

// ── Q4_K_M quantization ───────────────────────────────────────────────────────

// Pack one 256-element block into 144-byte block_q4_K format.
// Layout: [d:f16][dmin:f16][scales:u8×12][qs:u8×128]
// Dequant formula used by the kernel: d*sc*q - dmin*mn
fn quantize_block_q4k(vals: &[f32], block: &mut [u8]) {
    const N_SUB: usize = 8;
    const SUB_SIZE: usize = 32;

    let mut sub_scales = [0.0f32; N_SUB];
    let mut sub_mins   = [0.0f32; N_SUB];  // stored as positive (-actual_min)
    let mut sub_quants = [[0u8; SUB_SIZE]; N_SUB];

    for s in 0..N_SUB {
        let sub = &vals[s * SUB_SIZE..(s + 1) * SUB_SIZE];
        let actual_min = sub.iter().cloned().fold(f32::INFINITY, f32::min);
        let actual_max = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // clamp min to 0 so the offset term is always non-negative
        let min_val = actual_min.min(0.0);
        let scale = if actual_max > min_val {
            (actual_max - min_val) / 15.0
        } else {
            0.0
        };
        sub_scales[s] = scale;
        sub_mins[s] = -min_val;  // non-negative; kernel subtracts dmin*mn

        for i in 0..SUB_SIZE {
            let q = if scale > 0.0 {
                ((sub[i] - min_val) / scale + 0.5) as u8
            } else {
                0
            };
            sub_quants[s][i] = q.min(15);
        }
    }

    // Super-scales: quantize sub_scales and sub_mins to 6-bit
    let max_scale = sub_scales.iter().cloned().fold(0.0f32, f32::max);
    let max_min   = sub_mins.iter().cloned().fold(0.0f32, f32::max);

    let d    = max_scale / 63.0;
    let dmin = max_min   / 63.0;

    let mut sc6 = [0u8; N_SUB];
    let mut mn6 = [0u8; N_SUB];
    for s in 0..N_SUB {
        sc6[s] = if d    > 0.0 { (sub_scales[s] / d    + 0.5).min(63.0) as u8 } else { 0 };
        mn6[s] = if dmin > 0.0 { (sub_mins[s]   / dmin + 0.5).min(63.0) as u8 } else { 0 };
    }

    // Write d and dmin as f16 (little-endian)
    block[0..2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    block[2..4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());

    // Pack scales[12]: inverse of get_scale_min_k4 unpacking in the kernel
    let mut sb = [0u8; 12];
    for j in 0..4 {
        sb[j]     = sc6[j] & 63;
        sb[j + 4] = mn6[j] & 63;
    }
    for j in 4..8 {
        // lower 4 bits of sc6[j] → sb[j+4] bits 0-3
        // upper 2 bits of sc6[j] → sb[j-4] bits 6-7
        // lower 4 bits of mn6[j] → sb[j+4] bits 4-7
        // upper 2 bits of mn6[j] → sb[j]   bits 6-7
        sb[j + 4] |= sc6[j] & 0xF;
        sb[j - 4] |= (sc6[j] >> 4) << 6;
        sb[j + 4] |= (mn6[j] & 0xF) << 4;
        sb[j]     |= (mn6[j] >> 4) << 6;
    }
    block[4..16].copy_from_slice(&sb);

    // Pack qs[128]: two 4-bit quants per byte
    for s in 0..N_SUB {
        for i in (0..SUB_SIZE).step_by(2) {
            let byte_idx = (s * SUB_SIZE + i) / 2;
            block[16 + byte_idx] =
                (sub_quants[s][i] & 0xF) | ((sub_quants[s][i + 1] & 0xF) << 4);
        }
    }
}

// Quantize B matrix (stored as b_f16[batch*K*N + k*N + n]) to Q4_K_M.
// K must equal 256 (QK_K). One block_q4_K per (batch, column-n).
fn quantize_b_q4k(b_f16: &[f16], n_cols: usize, k_rows: usize, batch_size: usize) -> Vec<u8> {
    assert_eq!(k_rows, 256, "K must equal QK_K=256 for Q4_K_M");
    let mut out = vec![0u8; batch_size * n_cols * 144];

    for batch in 0..batch_size {
        for n in 0..n_cols {
            let vals: Vec<f32> = (0..k_rows)
                .map(|k| f32::from(b_f16[batch * k_rows * n_cols + k * n_cols + n]))
                .collect();
            let block_idx = batch * n_cols + n;
            quantize_block_q4k(&vals, &mut out[block_idx * 144..(block_idx + 1) * 144]);
        }
    }
    out
}

// ── Per-configuration OpenCL run ─────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_config(
    context: &Context,
    device_id: cl_device_id,
    source: &str,
    build_args: &str,
    at_u16: &[u16],
    b_q4k: &[u8],
    c_ref: &[f16],
    at_size: usize,
    c_size: usize,
    mdimc: usize,
    ndimc: usize,
    mwg: usize,
    nwg: usize,
) -> Result<(f32, f32), String> {
    let program = Program::create_and_build_from_source(context, source, build_args)
        .map_err(|e| format!("build: {:?}", e))?;
    let kernel =
        Kernel::create(&program, "HgemmBatched").map_err(|e| format!("kernel: {:?}", e))?;

    unsafe {
        let queue =
            CommandQueue::create_with_properties(context, device_id, CL_QUEUE_PROFILING_ENABLE, 0)
                .map_err(|e| format!("queue: {:?}", e))?;

        let mut a_buf =
            Buffer::<u16>::create(context, CL_MEM_READ_WRITE, at_size, std::ptr::null_mut())
                .map_err(|e| format!("buf_a: {:?}", e))?;
        // B is Q4K bytes, not u16 elements
        let mut b_buf =
            Buffer::<u8>::create(context, CL_MEM_READ_WRITE, b_q4k.len(), std::ptr::null_mut())
                .map_err(|e| format!("buf_b: {:?}", e))?;
        let mut c_buf =
            Buffer::<u16>::create(context, CL_MEM_READ_WRITE, c_size, std::ptr::null_mut())
                .map_err(|e| format!("buf_c: {:?}", e))?;

        queue
            .enqueue_write_buffer(&mut a_buf, 1, 0, at_u16, &[])
            .map_err(|e| format!("write_a: {:?}", e))?;
        queue
            .enqueue_write_buffer(&mut b_buf, 1, 0, b_q4k, &[])
            .map_err(|e| format!("write_b: {:?}", e))?;
        queue.finish().map_err(|e| format!("finish: {:?}", e))?;

        let local_x = 32 * mdimc / 16;
        let local_y = ndimc / 16;
        let global_x = 32 * M / 16 * mdimc / mwg;
        let global_y = N / 16 * ndimc / nwg;

        let m_arg = M as cl_int;
        let n_arg = N as cl_int;
        let k_arg = K as cl_int;

        let mut sum_time = 0u64;
        let mut sum_error = 0.0f32;
        let mut c_out = vec![0u16; c_size];

        for _ in 0..4 {
            let kern_ev = ExecuteKernel::new(&kernel)
                .set_arg(&m_arg)
                .set_arg(&n_arg)
                .set_arg(&k_arg)
                .set_arg(&a_buf)
                .set_arg(&b_buf)
                .set_arg(&c_buf)
                .set_global_work_sizes(&[global_x, global_y, BATCH_SIZE])
                .set_local_work_sizes(&[local_x, local_y, 1usize])
                .enqueue_nd_range(&queue)
                .map_err(|e| format!("enqueue: {:?}", e))?;

            queue.finish().map_err(|e| format!("finish2: {:?}", e))?;

            let t_start = kern_ev
                .profiling_command_start()
                .map_err(|e| format!("prof_start: {:?}", e))?;
            let t_end = kern_ev
                .profiling_command_end()
                .map_err(|e| format!("prof_end: {:?}", e))?;
            sum_time += t_end - t_start;

            queue
                .enqueue_read_buffer(&mut c_buf, 0, 0, &mut c_out, &[])
                .map_err(|e| format!("read_c: {:?}", e))?;
            queue.finish().map_err(|e| format!("finish3: {:?}", e))?;

            let c_f16: Vec<f16> = c_out.iter().map(|&v| f16::from_bits(v)).collect();
            sum_error += compare_ref(&c_f16, c_ref, N, M, BATCH_SIZE, N, M);
        }

        Ok((sum_time as f32 * 1e-6 / 4.0, sum_error / 4.0))
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string("hgemm.cl")
        .or_else(|_| fs::read_to_string("../hgemm.cl"))
        .map_err(|_| "hgemm.cl not found in current or parent directory")?;

    let platforms = get_platforms()?;
    println!("Detected {} OpenCL platforms.", platforms.len());

    let mut best_score = i32::MIN;
    let mut best_did: Option<cl_device_id> = None;
    let mut best_pname = String::new();
    let mut best_dname = String::new();
    let mut best_ver = 1.0f32;
    let mut dev_idx = 0u32;

    for p in &platforms {
        let pver = p.version().unwrap_or_default();
        let pprof = p.profile().unwrap_or_default();
        let pname = p.name().unwrap_or_default();
        let pvend = p.vendor().unwrap_or_default();
        println!("Platform version: {pver}");
        println!("Platform profile: {pprof}");
        println!("Platform name:    {pname}");
        println!("Platform vendor:  {pvend}");

        let ver = parse_opencl_version(&pver);
        for &did in &p.get_devices(CL_DEVICE_TYPE_ALL).unwrap_or_default() {
            let d = Device::new(did);
            let dname = d.name().unwrap_or_default();
            let dtype = d.dev_type().unwrap_or(0);
            let dvend = d.vendor().unwrap_or_default();
            let ddrv = d.driver_version().unwrap_or_default();
            let dclk = d.max_clock_frequency().unwrap_or(0);
            let dcu = d.max_compute_units().unwrap_or(0);

            println!("Device ID:     {dev_idx}");
            println!("Device name:   {}", dname.trim());
            println!("Device type:   {}", device_type_str(dtype));
            println!("Device vendor: {dvend}");
            println!("Device driver: {ddrv}");
            println!("Device speed:  {dclk} MHz");
            println!("Device cores:  {dcu} CU");

            let vl = dvend.to_lowercase();
            let mut score: i32 = 0;
            if vl.contains("advanced micro devices") || vl.contains("amd") {
                score += 1000;
            }
            if vl.contains("nvidia") {
                score += 1000;
            }
            if vl.contains("intel") {
                score += 500;
            }
            if dtype == CL_DEVICE_TYPE_GPU {
                score += 100;
            }
            score += (ver * 10.0) as i32;
            println!("Device score:  {score}");

            if score > best_score {
                best_score = score;
                best_did = Some(did);
                best_pname = pname.clone();
                best_dname = dname.trim().to_string();
                best_ver = ver;
            }
            dev_idx += 1;
        }
    }

    let device_id = best_did.ok_or("No suitable OpenCL device found")?;
    println!("\nSelected platform: {best_pname}");
    println!("Selected device:   {best_dname}");
    println!("with OpenCL {best_ver:.1} capability.\n");

    let device = Device::new(device_id);
    let context = Context::from_device(&device)?;

    // Pre-compute test data
    let m_max = M.max(64);
    let n_max = N.max(64);
    let k_max = K.max(32);

    let at_size = BATCH_SIZE * next_power_of_two(k_max) * next_power_of_two(m_max);
    let c_size  = BATCH_SIZE * next_power_of_two(m_max) * next_power_of_two(n_max);

    let mut at_f16 = vec![f16::ZERO; at_size];
    let mut b_f16  = vec![f16::ZERO; BATCH_SIZE * K * N];
    let mut c_ref  = vec![f16::ZERO; c_size];

    generate_data(&mut at_f16, K, M, BATCH_SIZE, K, M);
    generate_data(&mut b_f16,  N, K, BATCH_SIZE, N, K);
    reference_gemm(&at_f16, &b_f16, &mut c_ref, M, N, K, BATCH_SIZE);

    let at_u16: Vec<u16> = at_f16.iter().map(|v| v.to_bits()).collect();
    // Quantize B to Q4_K_M (one 144-byte block per column per batch)
    let b_q4k = quantize_b_q4k(&b_f16, N, K, BATCH_SIZE);
    println!("B matrix: {} f16 elements → {} Q4K bytes ({:.1}× compression)",
        BATCH_SIZE * K * N,
        b_q4k.len(),
        (BATCH_SIZE * K * N * 2) as f32 / b_q4k.len() as f32);

    // Auto-tuning loop
    let mut best_time = f32::MAX;
    let mut best_error = 0.0f32;
    let mut best_args = String::new();

    for &mdimc in &[16usize, 32, 64] {
        for &ndimc in &[16usize, 32] {
            for &mwg in &[16usize, 32, 64] {
                for &nwg in &[16usize, 32] {
                    if mwg < mdimc || nwg < ndimc {
                        continue;
                    }
                    for &kwg in &[16usize, 32] {
                        for sa in 0..=1usize {
                            for sb in 0..=1usize {
                                for &vwm in &[1usize, 2, 4, 8] {
                                    for &vwn in &[1usize, 2, 4, 8] {
                                        if sa == 0 && vwm != 1 {
                                            continue;
                                        }
                                        if sb == 0 && vwn != 1 {
                                            continue;
                                        }

                                        let tune = format!(
                                            " -DMDIMC={mdimc} -DNDIMC={ndimc} \
                                             -DMWG={mwg} -DNWG={nwg} -DKWG={kwg} \
                                             -DSA={sa} -DSB={sb} \
                                             -DVWM={vwm} -DVWN={vwn}"
                                        );
                                        let bargs = format!("{CL_BASE_ARGS}{tune}");

                                        match run_config(
                                            &context, device_id, &source, &bargs,
                                            &at_u16, &b_q4k, &c_ref,
                                            at_size, c_size,
                                            mdimc, ndimc, mwg, nwg,
                                        ) {
                                            Ok((time, error)) => {
                                                println!("{tune} {error:.6} {time:.3}");
                                                if time > 0.0 && time < best_time {
                                                    best_time = time;
                                                    best_error = error;
                                                    best_args = tune.clone();
                                                }
                                            }
                                            Err(e) => println!("{tune} FAILED: {e}"),
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("\n\nWinner: {best_args} {best_error:.6} {best_time:.3}");
    Ok(())
}
