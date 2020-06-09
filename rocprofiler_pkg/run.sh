#!/bin/bash
time_stamp=`date +%y%m%d_%H%M%S`

BIN_DIR1=`dirname $0`
pushd $BIN_DIR1 > /dev/null
BIN_DIR=$PWD
popd > /dev/null
RUN_DIR=$PWD
INPUT=$RUN_DIR/input.xml

hip_compile() {
  sample="$1"
  sample_src="src/${sample}"
  if [ ! -e "test/$sample" ] ; then
    cp -r /opt/rocm/hip/samples/*/*${sample} test/$sample_src
    make -C test/$sample_src > /dev/null 2>&1
    ln -sf ./$sample_src/$sample test/$sample
  fi
} 

mkdir -p test/src
#hip_compile hipEvent
hip_compile MatrixTranspose

apps_list_dflt="MatrixTranspose FastWalshTransform MatrixMultiplication SimpleConvolution StringSearch"
input_metrics_dflt="SQ_CYCLES, SQ_WAVES, SQ_INSTS_SMEM, SQ_INSTS_VALU, SQ_INSTS_VMEM_WR, TA_FLAT_WRITE_WAVEFRONTS[0], TA_FLAT_WRITE_WAVEFRONTS[1], CPC_ALWAYS_COUNT, CPC_ME1_STALL_WAIT_ON_RCIU_READ, GPUBusy, VALUBusy, SALUBusy, MemUnitBusy, SFetchInsts, FetchSize, VWriteInsts, WriteSize, TT"
output_dir_dflt=${time_stamp}_results

echo "Applications to run."
echo "HIP samples: MatrixTranspose"
echo "OpenCL samples: FastWalshTransform, MatrixMultiplication, SimpleConvolution, StringSearch"
read -p "Enter an applications list to run [all applicatinos by default]: " apps_list1
echo ""
echo "See 'metrics.xml' for supported profiling metrics."
read -p "Enter profiling metrics, enter 'TT' for thread trace [$input_metrics_dflt]: " input_metrics1
echo ""
echo "Profiling results will be dumped to files in a subdirectory or to terminal."
read -p "Enter output directory, output to terminal if '-' [$output_dir_dflt]: " output_dir
echo ""

if [ -z "$apps_list1" ] ; then
  apps_list1=$apps_list_dflt
fi

apps_list=`echo $apps_list1 | sed "s/[,:;]/ /g"`

if [ -z "$input_metrics1" ] ; then
  input_metrics1=$input_metrics_dflt
fi
input_metrics2=`echo "$input_metrics1" | sed -e "s/ \{1,\}/,/g" -e "s/   \{1,\}/,/g" -e "s/,\{1,\}/,/g" -e "s/,$//"`
input_metrics=`echo "$input_metrics2" | sed -e "s/,*TT//g"`

echo "<metric name=$input_metrics ></metric>" > $INPUT
if [ "$input_metrics" != "$input_metrics2" ] ; then
  cat >> $INPUT <<EOF
<trace name="SQTT">
  <parameters
    TARGET_CU=0
  ></parameters>
</trace>
EOF
fi

if [ -z "$output_dir" ] ; then
  output_dir=$output_dir_dflt
fi
if [ "$output_dir" = "-" ] ; then
  output_dir=""
else
  if [ -e "$output_dir" ] ; then
    echo "Output directory '$output_dir' exists"
    read -p "Do you want to remove '$output_dir' [Y/N]: " answer
    if [ "$answer" = "Y" ] ; then
      rm -rf "$output_dir"
    else
      echo "Exiting"
      exit 1
    fi
  fi
  mkdir $output_dir
fi

ROCP_OUTPUT_DIR=""

cd test
for name in $apps_list ; do
  if [ -n "$output_dir" ] ; then
    ROCP_OUTPUT_DIR=$RUN_DIR/$output_dir/$name
    output_file=$ROCP_OUTPUT_DIR/output.csv
  else
    ROCP_OUTPUT_DIR=$RUN_DIR/$output_dir_dflt/$name
    output_file="--"
  fi
  echo ""
  $BIN_DIR/rpl_run.sh --timestamp on -i $INPUT -o $output_file -d $ROCP_OUTPUT_DIR ./$name
done

exit 0
