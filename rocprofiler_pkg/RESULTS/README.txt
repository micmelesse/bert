'inst' group:

Wavefronts : Total wavefronts.
    Wavefronts = SQ_WAVES
VALUInsts : The average number of vector ALU instructions executed per work-item (affected by flow control).
    VALUInsts = (SQ_INSTS_VALU / SQ_WAVES)
SALUInsts : The average number of scalar ALU instructions executed per work-item (affected by flow control).
    SALUInsts = (SQ_INSTS_SALU / SQ_WAVES)
SFetchInsts : The average number of scalar fetch instructions from the video memory executed per work-item (affected by flow control).
    SFetchInsts = (SQ_INSTS_SMEM / SQ_WAVES)
FlatVMemInsts : The average number of FLAT instructions that read from or write to the video memory executed per work item (affected by flow control). Includes FLAT instructions that read from or write to scratch.
    FlatVMemInsts = ((SQ_INSTS_FLAT - SQ_INSTS_FLAT_LDS_ONLY) / SQ_WAVES)
LDSInsts : The average number of LDS read or LDS write instructions executed per work item (affected by flow control).  Excludes FLAT instructions that read from or write to LDS.
    LDSInsts = ((SQ_INSTS_LDS - SQ_INSTS_FLAT_LDS_ONLY) / SQ_WAVES)
FlatLDSInsts : The average number of FLAT instructions that read or write to LDS executed per work item (affected by flow control).
    FlatLDSInsts = (SQ_INSTS_FLAT_LDS_ONLY / SQ_WAVES)
GDSInsts : The average number of GDS read or GDS write instructions executed per work item (affected by flow control).
    GDSInsts = (SQ_INSTS_GDS / SQ_WAVES)
VALUUtilization : The percentage of active vector ALU threads in a wave. A lower number can mean either more thread divergence in a wave or that the work-group size is not a multiple of 64. Value range: 0% (bad), 100% (ideal - no thread divergence).
    VALUUtilization = ((100 * SQ_THREAD_CYCLES_VALU) / (SQ_ACTIVE_INST_VALU * 64))
FetchSize : The total kilobytes fetched from the video memory. This is measured with all extra fetches and any cache or memory effects taken into account.
    FetchSize = ((((sum(TCC_EA_RDREQ,16) - sum(TCC_EA_RDREQ_32B,16)) * 64) + (sum(TCC_EA_RDREQ_32B,16) * 32)) / 1024)

'inst1' group:

WriteSize : The total kilobytes written to the video memory. This is measured with all extra fetches and any cache or memory effects taken into account.
    WriteSize = ((((sum(TCC_EA_WRREQ,16) - sum(TCC_EA_WRREQ_64B,16)) * 32) + (sum(TCC_EA_WRREQ_64B,16) * 64)) / 1024)
L2CacheHit : The percentage of fetch, write, atomic, and other instructions that hit the data in L2 cache. Value range: 0% (no hit) to 100% (optimal).
    L2CacheHit = ((100 * sum(TCC_HIT,16)) / (sum(TCC_HIT,16) + sum(TCC_MISS,16)))

'mem' group:

VFetchInsts : The average number of vector fetch instructions from the video memory executed per work-item (affected by flow control). Excludes FLAT instructions that fetch from video memory.
    VFetchInsts = (SQ_INSTS_VMEM_RD-TA_FLAT_READ_WAVEFRONTS_sum)/SQ_WAVES
VWriteInsts : The average number of vector write instructions to the video memory executed per work-item (affected by flow control). Excludes FLAT instructions that write to video memory.
    VWriteInsts = (SQ_INSTS_VMEM_WR-TA_FLAT_WRITE_WAVEFRONTS_sum)/SQ_WAVES

'util' group:

GPUBusy : The percentage of time GPU was busy.
    GPUBusy = ((100 * GRBM_GUI_ACTIVE) / GRBM_COUNT)
VALUBusy : The percentage of GPUTime vector ALU instructions are processed. Value range: 0% (bad) to 100% (optimal).
    VALUBusy = ((((100 * SQ_ACTIVE_INST_VALU) * 4) / 64) / GRBM_GUI_ACTIVE)
SALUBusy : The percentage of GPUTime scalar ALU instructions are processed. Value range: 0% (bad) to 100% (optimal).
    SALUBusy = ((((100 * SQ_INST_CYCLES_SALU) * 4) / 64) / GRBM_GUI_ACTIVE)
MemUnitStalled : The percentage of GPUTime the memory unit is stalled. Try reducing the number or size of fetches and writes if possible. Value range: 0% (optimal) to 100% (bad).
    MemUnitStalled = (((100 * max(TCP_TA_DATA_STALL_CYCLES,16)) / GRBM_GUI_ACTIVE) / 4)
WriteUnitStalled : The percentage of GPUTime the Write unit is stalled. Value range: 0% to 100% (bad).
    WriteUnitStalled = ((100 * max(TCC_EA_WRREQ_STALL,16)) / GRBM_GUI_ACTIVE)
LDSBankConflict : The percentage of GPUTime LDS is stalled by bank conflicts. Value range: 0% (optimal) to 100% (bad).
    LDSBankConflict = (((100 * SQ_LDS_BANK_CONFLICT) / GRBM_GUI_ACTIVE) / 64)

'mem_util' group:

MemUnitBusy : The percentage of GPUTime the memory unit is active. The result includes the stall time (MemUnitStalled). This is measured with all extra fetches and writes and any cache or memory effects taken into account. Value range: 0% to 100% (fetch-bound).
    MemUnitBusy = (((100 * max(TA_BUSY,16)) / GRBM_GUI_ACTIVE) / 4)
