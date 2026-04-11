``` bash
sakib@sakib:~/codes/competitions/dc_ops/training/data$ python generate_sft_data.py 
Generating seed episodes...
  Seed episodes: 8
  Command drills: 26
Augmenting with reasoning variations...
  Reasoning augmentations: 24
Augmenting with setpoint variations...
  Setpoint augmentations: 6

Total episodes: 64
Output: dc_ops_sft_data.jsonl
Total conversation turns: 636
Agent turns (training targets): 286
Unique command types used: ['acknowledge_alarm', 'adjust_setpoint', 'check_status', 'diagnose', 'escalate', 'refuel_generator', 'set_fan_speed', 'set_rack_load', 'set_ups_mode', 'start_crac', 'start_generator', 'stop_crac', 'stop_generator', 'wait']
```
---
``` bash
sakib@sakib:~/codes/competitions/dc_ops/training/data$ python augment_sft_data.py
Loaded 64 seed episodes
Generating A1 (Setpoint Optimization) variations...
  A1 episodes: 105 = 105
Generating A2 (Thermal Event) variations...
  A2 episodes: ~42
Generating A4 (CRAC Cascade) variations...
  A4 episodes: 20
Generating B1 (UPS Alarm) variations...
  B1 episodes: 30
Generating B3 (Generator Test) variations...
  B3 episodes: 25
Generating B4 (Power Failure) variations...
  B4 episodes: 45

==================================================
TOTAL EPISODES: 331
Output: dc_ops_sft_augmented.jsonl
Total conversation turns: 5379
Agent turns (training targets): 2524

Command distribution:
  adjust_setpoint             667
  diagnose                    441
  set_fan_speed               418
  wait                        401
  check_status                319
  set_rack_load               146
  acknowledge_alarm            64
  start_generator              30
  stop_generator               30
  set_ups_mode                  3
  refuel_generator              2
  escalate                      1
  stop_crac                     1
  start_crac                    1

File size: 11.0 MB
```
---
``` bash
sakib@sakib:~/codes/competitions/dc_ops/training/data$ python add_underrepresented.py
Existing episodes: 331
Generating UPS mode episodes...
  UPS mode: 18
Generating refuel episodes...
  Refuel: 6
Generating start/stop CRAC episodes...
  Start/Stop CRAC: 24
Generating escalation episodes...
  Escalation: 12
Generating load shedding episodes...
  Load shedding: 50
Generating diagnose-all-targets episodes...
  Diagnose targets: 51
Generating multi-step setpoint+fan episodes...
  Multi-step: 40

==================================================
TOTAL EPISODES: 532
Output: dc_ops_sft_final.jsonl
Total conversation turns: 6142
Agent turns (training targets): 2805

Command distribution:
  adjust_setpoint             710  (25.3%)
  diagnose                    495  (17.6%)
  set_fan_speed               458  (16.3%)
  wait                        401  (14.3%)
  check_status                359  (12.8%)
  set_rack_load               196  (7.0%)
  acknowledge_alarm            64  (2.3%)
  start_generator              33  (1.2%)
  stop_generator               30  (1.1%)
  set_ups_mode                 21  (0.7%)
  start_crac                   13  (0.5%)
  stop_crac                    13  (0.5%)
  refuel_generator              8  (0.3%)
  escalate                      4  (0.1%)

File size: 11.5 MB
```

