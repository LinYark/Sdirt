
<div align="center">
  <h2><strong>Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation</strong></h2>
  <p>
    <a href="https://github.com/LinYark" target="_blank" rel="noopener noreferrer">Fengchen He</a>, Dayang Zhao, Hao Xu, Tingwei Quan, Shaoqun Zeng<br>
    HUST, China
  </p>

  <p>
    <a href="https://arxiv.org/abs/2503.11213" target="_blank" rel="noopener noreferrer">
      📚 arXiv
    </a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#" target="_blank" rel="noopener noreferrer">
      📄 Paper
    </a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#" target="_blank" rel="noopener noreferrer">
      🧾 Supp
    </a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <!-- <a href="#" target="_blank" rel="noopener noreferrer">
      📦 Dataset
    </a> &nbsp;&nbsp;|&nbsp;&nbsp; -->
    <a href="#" target="_blank" rel="noopener noreferrer">
      🔗 Project
    </a>
  </p>
</div>


## 🗓️ News

- **[2025-07]** Camera-ready version done; open-source repo & website coming in August 2025. 🚧 
- **[2025-06]** Our paper **Sdirt** has been accepted to **ICCV 2025** 🎉, see you in Hawaii! 🌺
- **[2025-04]** This is the official repository of Sdirt. 📘


# TL;DR
Dual-Pixel (DP) images are valuable for depth estimation, but real DP-depth paired datasets are scarce.  
**Sdirt** leverages **ray tracing** to simulate realistic DP images, effectively reducing the **domain gap** between synthetic and real data.

# Abstract
Many studies utilize dual-pixel (DP) sensor phase information for various applications, such as depth estimation and deblurring. However, since DP image features are entirely determined by the camera hardware, DP-depth paired datasets are very scarce, especially when performing depth estimation on customized cameras. To overcome this, studies simulate DP images using ideal optical models. However, these simulations often violate real optical propagation laws, leading to poor generalization to real DP data. To address this, we investigate the domain gap between simulated and real DP data, and propose solutions using the Simulating DP Images from Ray Tracing (Sdirt) scheme. Sdirt generates realistic DP images via ray tracing and integrates them into the depth estimation training pipeline. Experimental results show that models trained with Sdirt-simulated images generalize better to real DP data.

<div align="center">
  <img src="images/main.png" alt="Sdirt Overview" width="90%">
</div>

---
# 🚀 Getting Started
To learn more usage about this codebase, kindly refer to [GET_START.md](./docs/GET_START.md).

# 🛠️ Code & Dataset Release Plan
- ✅ Paper available on arXiv with citation examples
- 🚧 Open-source Sdirt in stages:
  - [ ] Demo
  - [ ] Dataset
  - [ ] Full source code
- 🔐 Full code will be released **after the paper is officially published**



# Citations
We appreciate a star ⭐ if you'd like to follow future updates.
If you find it useful, please consider citing our paper:
```bibtex
@article{he2025simulating,
  title={Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation},
  author={He, Fengchen and Zhao, Dayang and Xu, Hao and Quan, Tingwei and Zeng, Shaoqun},
  journal={arXiv preprint arXiv:2503.11213},
  year={2025}
}
```

# Acknowledgments
This work was supported by National Natural Science Foundation of China (Grant No. 32471146) and the project N20240194.
The authors thank Echossom, Miya, and Xinge for valuable discussions and assistance.

---
🤔 Btw, I am seeking help from any engineer familiar with Dual-Pixel sensors. (⚡plz contact [me](https://github.com/LinYark), crying⚡).
<p align="right">
  <a href="https://github.com/LinYark/Sdirt">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=LinYark.Sdirt" alt="visitors"/>
  </a>
</p>
<!-- [![visitors](https://visitor-badge.laobi.icu/badge?page_id=LinYark.Sdirt)](https://github.com/LinYark/Sdirt) -->