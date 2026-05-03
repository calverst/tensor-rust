# hgemmtest port to rust with Q4_K_M
the original code is located at https://github.com/ihavnoid/hgemmtest
100% vibe code port to rust using Sonnet 4.6. Eigen librarly was removed.
### Some test results
#### Original winner:
  -DMDIMC=16 -DNDIMC=16 -DMWG=16 -DNWG=32 -DKWG=32 -DSA=0 -DSB=0 -DVWM=1 -DVWN=1
  error = 0.000423   time = 0.013 ms
  No local memory caching (SA=0 SB=0) wins — expected for a small N=32 problem where tensor cores saturate global bandwidth before shared memory staging pays off.
#### After Q4_K_M is added:
-DMDIMC=16 -DNDIMC=16 -DMWG=16 -DNWG=16 -DKWG=32 -DSA=0 -DSB=1 -DVWM=1 -DVWN=8
MSE 0.002888   time 0.045 ms
Best config uses SB=1 (local B staging), which is required for Q4K dequantization, so it naturally wins



