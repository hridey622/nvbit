from nvbpf_py import array, gemm_wavefit, percpu_array, tool


@tool(
    "blackwell_persistent_balance_py",
    banner="BLACKWELL_PERSISTENT_BALANCE_PY",
)
class BlackwellPersistentBalancePy:
    sm_cta_entries = percpu_array(
        type_name="u64",
        length=1,
        description="Per-SM CTA entry count proxy for persistent scheduler balance.",
    )
    active_sm_bitmap = array(
        type_name="u64",
        length=4,
        description="Bitmap of SMs touched by the launch.",
    )
    analysis = gemm_wavefit(
        sm_cta_entries_map="sm_cta_entries",
        active_sm_bitmap_map="active_sm_bitmap",
        filter_csv="cutlass,gemm,tcgen05,mma,attention,persistent",
        filter_env="NVBPF_KERNEL_FILTER",
    )
