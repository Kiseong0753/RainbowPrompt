# RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning

[Official PyTorch code]

**[RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning](https://www.arxiv.org/abs/2507.22553)**  
*Kiseong Hong, Gyeong-hyeon Kim, Eunwoo Kim*  
IEEE/CVF International Conference on Computer Vision (ICCV), 2025


![RainbowPrompt Overview](./Overview.png)
---

## Abstract
Prompt-based continual learning provides a rehearsal-free solution by tuning small sets of parameters while keeping pre-trained models frozen. To meet the complex demands of sequential tasks, it is crucial to integrate task-specific knowledge within prompts effectively. However, existing works rely on either fixed learned prompts (i.e., prompts whose representations remain unchanged during new task learning) or on prompts generated from an entangled task-shared space, limiting the representational diversity of the integrated prompt. To address this issue, we propose a novel prompt-evolving mechanism to adaptively aggregate base prompts (i.e., task-specific prompts) into a unified prompt while ensuring diversity. By transforming and aligning base prompts, both previously learned and newly introduced, our approach continuously evolves accumulated knowledge to facilitate learning new tasks. We further introduce a learnable probabilistic gate that adaptively determines which layers to activate during the evolution process. We validate our method on image classification and video action recognition tasks in class-incremental learning, achieving average gains of 9.07% and 7.40% over existing methods across all scenarios.
---

