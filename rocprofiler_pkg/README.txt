The package provides compute specific ROCm profiling library (ROC profiler) and a profiling tool.
ROC profiler library provides GPU performance metrics, traces and events API.
Provided tool is based on the library and let you to dump per kernel performance counters and thread traces data.
Performance counters are dumped to 'results.txt' file and the thread traces to per shader 'thread_trace_<n>__<kernel-name>_se<ind>.out' files with ThreadTraceViewer data format.
GFX8 Ellesmere/Buffin/Fiji and GFX9 Vega10/Vega20 are supported.

ROCm 1.8 or the latest internal ROCm build should be installed.
The latest AQL profile HSA extension version is provided with the package.

 - run.sh - interactive demo run script to run OpenCL and HIP sampoles and dumping some metrics
 - rpl_run.sh - script to run an application with profiling
   $ rpl_run.sh [-h] [--list-basic] [--list-derived] [-i <input .txt/.xml file>] [-o <output CSV file>] <app command line>

 - run_papi.sh - run simple PAPI test

 - RESULTS - subdirectory with input .txt and .xml files for 5 derived metric groups 'inst', inst1', 'mem', 'util', 'mem_util' and README.txt with metrics descriptions and formulas

 - ThreadTraceView
   - ThreadTraceView - Thread Trace viewer tool for Linux
   - config.ini - Thread Trace viewer config file

 - tool :
   - libtool.so - test tool library
   - tool.cpp - sources of the test tool library
   - papi_* - PAPI utilities

 - lib :
   - librocprofiler64.so - ROC Profiler library
   - metrics.xml - ROC Profiler metrics
   - hsa
     - libhsa-amd-aqlprofile64.so.1 - extended aqlprofile HSA extrension version to support ROC Profiler
   - papi - PAPI library

 - inc :
   - rocprofiler.h - library API header

 - test :
   - SimpleConvolution - simple-convolution OpenCL sample
   - StringSearch - string-search OpenCL sample
   - FastWalshTransform - fsat-walch-transform OpenCL sample
   - hipEvent_gfx8/gfx9 - hipEvent gfx8/gfx9 HIP samples
   - MatrixTranspose_gfx8/gfx9 - MatrixTranspose gfx8/gfx9 HIP samples

 - doc - documentation

User instructions

1. Download and unpack the package "rocprofiler_pkg_<rev>.tgz". Please find the link at the bottom of this page in documentation section.
2. To collect per dispatch counters and thread traces you can use 'rpl_run.sh' script from the package. To run the script you can use either explicit path to the package or set PATH env var.

    $ rpl_run.sh [-h] [--list-basic] [--list-derived] [-i <input .txt/.xml file>] [-o <output CSV file>] <app command line>

    Options:
      -h - this help
      --verbose - verbose mode, dumping all base counters used in the input derived metrics
      --list-basic - to print the list of basic HW counters
      --list-derived - to print the list of derived metrics with formulas

      -i <.txt|.xml file> - input file
          Input file .txt format, automatically rerun application for every pmc/sqtt line:
    
            # Filter by dispatches range, GPU index and kernel names
            # supported range formats: "3:9", "3:", "3"
            range: 1 : 4
            gpu: 0 1 2 3
            kernel: Simple1 Conv1 SimpleConvolution
            # Perf counters group 1
            pmc : Wavefronts VALUInsts SALUInsts SFetchInsts FlatVMemInsts LDSInsts FlatLDSInsts GDSInsts VALUUtilization FetchSize
            # Perf counters group 2
            pmc : WriteSize L2CacheHit
            # SQ tread trace
            sqtt : MASK = 0x0F00 TOKEN_MASK = 0x144B TOKEN_MASK2 = 0xFFFF
    
          Input file .xml format, for single profiling run:
    
            # Filter by dispatches range, GPU index and kernel names
            <metric
              # range formats: "3:9", "3:", "3"
              range=""
              # list of gpu indexes "0,1,2,3"
              gpu_index=""
              # list of matched sub-strings "Simple1,Conv1,SimpleConvolution"
              kernel=""
            ></metric>
    
            # Metrics list definition, also the form "<block-name>:<event-id>" can be used
            # All defined metrics can be found in the 'metrics.xml'
            # There are basic metrics for raw HW counters and high-level metrics for derived counters
            <metric name=SQ:4,SQ_WAVES,CPC_ALWAYS_COUNT,VFetchInsts
            ></metric>
    
            # Trace enabling and the parameters definition
            <trace name="SQTT">
              <parameters
                MASK=0x0F00
                TOKEN_MASK=0x144B
                TOKEN_MASK2=0xFFFF
              ></parameters>
            </trace>

            Supported by profiler SQTT parameters:
              SE_MASK - mask of traced Shader Engines, valid masks are: 0x1, 0x3, 0x7, 0xf
              TARGET_CU - target Compute Unit, MASK.CU_SEL field
              VM_ID_MASK - select which VM IDs to capture, MASK.VM_ID_MASK field
              MASK - MASK register value
              TOKEN_MASK - TOKEN_MASK register value
              TOKEN_MASK2 - TOKEN_MASK2 register value, traced instructions mask
            The parameters defaults:
              SE_MASK = <all enabled>
              TARGET_CU = 0;
              VM_ID_MASK = 0;
              MASK:
                  mask.bits.CU_SEL = param{TARGET_CU};
                  mask.bits.SH_SEL = 0x0;
                  mask.bits.SIMD_EN = 0xF;
                  mask.bits.SQ_STALL_EN = 0x1;
                  mask.bits.SPI_STALL_EN = 0x1;
                  mask.bits.REG_STALL_EN = 0x1;
                  mask.bits.VM_ID_MASK = param{VM_ID_MASK};
              TOKEN_MASK:
                  token_mask.bits.TOKEN_MASK = 0xFFFF;
                  token_mask.bits.REG_MASK = 0xFF;
                  token_mask.bits.REG_DROP_ON_STALL = 0x1;
              TOKEN_MASK2:
                  token_mask2.bits.INST_MASK = 0xFFFFFF7F; // INST_PC is disabled because its tracing can cause extra stalling
                                                           // and it is recommended to disable by SQTT user guide
              HIWATER = 6; // which is 6/8 fraction of the tread trace fifo
    
      -o <output file> - output CSV file [<input file base>.csv]
      -d <data directory> - directory where profiler store profiling data including thread treaces [/tmp]
          The data directory is renoving autonatically if the directory is matching the temporary one, which is the default.
      -t <temporary directory> - to change the temporary directory [/tmp]
          By changing the temporary directory you can prevent removing the profiling data from /tmp or enable removing from not '/tmp' directory.

      --basenames <on|off> - to turn on/off truncating of the kernel full function names till the base ones [off]
      --timestamp <on|off> - to turn on/off the kernel dispatches timestamps, dispatch/begin/end/complete [off]
      --ctx-limit <max number> - maximum number of outstanding contexts [0 - unlimited]
      --heartbeat <rate sec> - to print progress heartbeats [0 - disabled]
      --sqtt-size <byte size> - to set SQTT buffer size, aggregate for all SE [0x2000000]
          Can be set in KB (1024B) or MB (1048576) units, examples 20K or 20M respectively.
      --sqtt-local <on|off> - to allocate SQTT buffer in local GPU memory [on]

    Configuration file:
      You can set your defaults preferences in the configuration file 'rpl_rc.xml'. The search path sequence: .:${HOME}:<package path>
      First the configuration file is looking in the current directory, then in your home, and then in the package directory.
      Configurable options: 'basenames', 'timestamp', 'ctx-limit', 'heartbeat', 'sqtt-size', 'sqtt-local'.
      An example of 'rpl_rc.xml':
        <defaults
          basenames=off
          timestamp=off
          ctx-limit=0
          heartbeat=0
          sqtt-size=0x20M
          sqtt-local=on
        ></defaults>

Interactive script
Below is the log for 'SimpleConvolution' OpenCL sample:

$ ./run.sh
Applications to run.
HIP samples: hipEvent_gfx8, hipEvent_gfx9, MatrixTranspose_gfx8, MatrixTranspose_gfx9
OpenCL samples: FastWalshTransform, MatrixMultiplication, SimpleConvolution, StringSearch
Enter an applications list to run [all applicatinos by default]: SimpleConvolution

See 'metrics.xml' for supported profiling metrics.
Enter profiling metrics, enter 'TT' for thread trace [SQ_CYCLES, SQ_WAVES, CPC_ALWAYS_COUNT, CPC_ME1_STALL_WAIT_ON_RCIU_READ, VALU_INSTS, SFETCH_INSTS, VWRITE_INSTS, FETCH_SIZE, TA_FLAT_WRITE_WAVEFRONTS[0], TA_FLAT_WRITE_WAVEFRONTS[1], TT]: 

Profiling results will be dumped to files in a subdirectory or to terminal.
Enter output directory, output to terminal if '-' [171130_141618_results]: -
Profiling 'SimpleConvolution'
ROCProfiler: input from "/home/evgeny/src/rocprofiler_pkg/input.xml"
  10 metrics
    SQ_CYCLES, SQ_WAVES, CPC_ALWAYS_COUNT, CPC_ME1_STALL_WAIT_ON_RCIU_READ, VALU_INSTS, SFETCH_INSTS, VWRITE_INSTS, FETCH_SIZE, TA_FLAT_WRITE_WAVEFRONTS[0], TA_FLAT_WRITE_WAVEFRONTS[1]
  1 traces
    SQTT (
      HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK = 0xf
      HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK = 0xf
    )
HsaProxyQueue::Init()
Platform 0 : Advanced Micro Devices, Inc.
Platform found : Advanced Micro Devices, Inc.

Selected Platform Vendor : Advanced Micro Devices, Inc.
Device 0 : gfx803 Device ID is 0x2141e10
HsaProxyQueue::Init()
Executing kernel for 1 iterations
-------------------------------------------
ROCProfiler: importing metrics from '/home/evgeny/src/rocprofiler_pkg/metrics.xml':
  29 gfx8 metrics found
  6 global metrics found

ROCPRofiler: 3 contexts collected
Dispatch[0], queue_index(0), kernel_object(0x901d0c000), kernel_name("simpleNonSeparableConvolution"):
  SQ_CYCLES (359792)
  SQ_WAVES (4096)
  CPC_ALWAYS_COUNT (89947)
  CPC_ME1_STALL_WAIT_ON_RCIU_READ (5513)
  VALU_INSTS (242)
  SFETCH_INSTS (17)
  VWRITE_INSTS (0)
  FETCH_SIZE (1034)
  TA_FLAT_WRITE_WAVEFRONTS[0] (460)
  TA_FLAT_WRITE_WAVEFRONTS[1] (460)
  SQTT (
    SE(0) size(11936)
    SE(1) size(8832)
    SE(2) size(10752)
    SE(3) size(9216)
  )
Dispatch[1], queue_index(2), kernel_object(0x901d0c400), kernel_name("simpleSeparableConvolutionPass1"):
  SQ_CYCLES (377548)
  SQ_WAVES (4112)
  CPC_ALWAYS_COUNT (94386)
  CPC_ME1_STALL_WAIT_ON_RCIU_READ (5566)
  VALU_INSTS (93)
  SFETCH_INSTS (11)
  VWRITE_INSTS (0)
  FETCH_SIZE (1034)
  TA_FLAT_WRITE_WAVEFRONTS[0] (456)
  TA_FLAT_WRITE_WAVEFRONTS[1] (448)
  SQTT (
    SE(0) size(13600)
    SE(1) size(12032)
    SE(2) size(12768)
    SE(3) size(12160)
  )
Dispatch[2], queue_index(4), kernel_object(0x901d0c700), kernel_name("simpleSeparableConvolutionPass2"):
  SQ_CYCLES (954336)
  SQ_WAVES (4096)
  CPC_ALWAYS_COUNT (238583)
  CPC_ME1_STALL_WAIT_ON_RCIU_READ (8071)
  VALU_INSTS (92)
  SFETCH_INSTS (11)
  VWRITE_INSTS (0)
  FETCH_SIZE (1032)
  TA_FLAT_WRITE_WAVEFRONTS[0] (440)
  TA_FLAT_WRITE_WAVEFRONTS[1] (432)
  SQTT (
    SE(0) size(13664)
    SE(1) size(12512)
    SE(2) size(13088)
    SE(3) size(12160)
  )
