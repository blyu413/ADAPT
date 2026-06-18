# ADAPT: Analytical Disturbance-Aware Policy Training for Humanoid Locomotion

Project page: https://blyu413.github.io/adapt-locomotion/

Paper: https://arxiv.org/abs/2606.16542

![ADAPT overview](images/overview.png)

## Abstract

Humanoids deployed in human-centered environments must handle force-interactive tasks, where external contacts introduce unexpected disturbances that disrupt locomotion accuracy and stability. Existing learning-based approaches rely on broad domain randomization, task-specific force objectives, or learning-based force estimators from motion history, each of which compromises accuracy, task transferability, or out-of-distribution robustness. We present Analytical Disturbance-Aware Policy Training (ADAPT), a framework that equips humanoid policies with a physically grounded disturbance observer. ADAPT estimates residual force/torque online from accessible robot dynamics without requiring force/torque sensors, feeds these estimates directly into the policy, and uses them as a physics-derived signal for robust behavior under external forces, payloads, and contact-induced disturbances. Experiments on a Unitree G1 humanoid show accurate disturbance prediction, stronger robustness than a proprioception-only baseline, improved velocity tracking under out-of-distribution disturbances, and the ability to encourage lighter locomotion by penalizing inferred lower-body disturbances.

## Citation

```bibtex
@misc{lyu2026adaptanalyticaldisturbanceawarepolicy,
  title={ADAPT: Analytical Disturbance-Aware Policy Training for Humanoid Locomotion},
  author={Bofan Lyu and Jindou Jia and Kuangji Zuo and Yanshuo Lu and Shijia Han and Gen Li and Boyu Ma and Jingliang Li and Geng Li and Jianfei Yang},
  year={2026},
  eprint={2606.16542},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```
