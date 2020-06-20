# step to downgrade ROCm thunk to ROCm3.3

wget http://repo.radeon.com/rocm/apt/.apt_3.3/pool/main/h/hsakmt-roct/hsakmt-roct_1.0.9-330-gd84bc09_amd64.deb
wget http://repo.radeon.com/rocm/apt/.apt_3.3/pool/main/h/hsakmt-roct-dev/hsakmt-roct-dev_1.0.9-330-gd84bc09_amd64.deb
dpkg-deb -vx hsakmt-roct-dev_1.0.9-330-gd84bc09_amd64.deb .
dpkg-deb -vx hsakmt-roct_1.0.9-330-gd84bc09_amd64.deb .
# rm -rf /opt/rocm-3.5.0/
# cp -r opt/rocm-3.3.0/* /opt/rocm-3.3.0