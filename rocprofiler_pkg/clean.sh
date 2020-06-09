#!/bin/sh -x
rm -rf *_results
rm -rf *_results_*
rm -rf */*_results_*
rm -f *.csv
rm -f */*.csv
rm -rf rpl_data_*
rm -rf */rpl_data_*
rm -f input.xml
rm -rf test/src

cat > ThreadTraceView/config.ini << EOF
[View]
TrackView=0
TrackViewCompact=0
TrackViewClamp=0
TrackViewColorShader=0
GraphView=1
CounterView=0
CounterColor=0
SpmDrawView=0
Track=1
Graph=1
Counter=0
SpmDraw=0
[ThreadTrace]
[ThreadTrace/Event]
LostPacket=1
EOF
