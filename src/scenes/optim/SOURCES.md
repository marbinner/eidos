# Sources of the Optimisation course

The scenes in this directory and the two libraries they stand on
(`src/math/optim.rs`, `src/math/terrain.rs`) take every statement about published
work, and every rule's exact form, from the sources below. Comments in the code name
the source first (authors, year, equation or section); some add a pointer such as
"course notes, noise.md D2". Those notes are the working files in which each source
was read and checked before the course was written (four files: `muon.md`,
`noise.md`, `curvature.md`, `classics.md`). They are not part of this repository;
this list is.

A venue is given only where the source's own page states it, and authors are named
as fully as the notes recorded them. Preprints are marked by their arXiv number
alone.

## Step sizes and momentum on a bowl
- G. Goh, "Why Momentum Really Works", Distill 2(4):e6, 2017.
- PyTorch documentation (version 2.15): `torch.optim.SGD`, `Adagrad`, `RMSprop`,
  `Adam`, `AdamW`, `Muon`.

## Adam, sign descent and their relatives
- D. P. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015,
  arXiv:1412.6980.
- Reddi, Kale, Kumar, "On the Convergence of Adam and Beyond", ICLR 2018,
  arXiv:1904.09237.
- I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization", ICLR 2019,
  arXiv:1711.05101.
- L. Balles, P. Hennig, "Dissecting Adam: The Sign, Magnitude and Variance of
  Stochastic Gradients", ICML 2018, arXiv:1705.07774.
- J. Bernstein, Y.-X. Wang, K. Azizzadenesheli, A. Anandkumar, "signSGD: Compressed
  Optimisation for Non-Convex Problems", arXiv:1802.04434.
- Karimireddy, Rebjock, Stich, Jaggi, "Error Feedback Fixes SignSGD
  and other Gradient Compression Schemes", ICML 2019, arXiv:1901.09847.
- X. Chen, C. Liang, D. Huang and co-authors, "Symbolic Discovery of Optimization
  Algorithms" (Lion), arXiv:2302.06675.
- S. Malladi, K. Lyu, A. Panigrahi, S. Arora, "On the SDEs and Scaling Rules for
  Adaptive Gradient Algorithms", arXiv:2205.10287.
- F. Kunstner, J. Chen, J. W. Lavington, M. Schmidt, "Noise Is Not the Main Factor
  Behind the Gap Between SGD and Adam on Transformers, but Sign Descent Might Be",
  arXiv:2304.13960.
- Kunstner, Yadav, Milligan, Schmidt, Bietti, "Heavy-Tailed Class
  Imbalance and Why Adam Outperforms Gradient Descent on Language Models",
  arXiv:2402.19449.
- J. Zhang, S. P. Karimireddy, A. Veit, S. Kim, S. J. Reddi, S. Kumar, S. Sra, "Why
  are Adaptive Methods Good for Attention Models?", arXiv:1912.03194.

## Mini-batch noise, the noise scale and the critical batch size
- S. McCandlish, J. Kaplan, D. Amodei and the OpenAI Dota Team, "An Empirical Model
  of Large-Batch Training", arXiv:1812.06162.
- S. L. Smith, Q. V. Le, "A Bayesian Perspective on Generalization and Stochastic
  Gradient Descent", ICLR 2018, arXiv:1710.06451.
- S. L. Smith, P.-J. Kindermans, C. Ying, Q. V. Le, "Don't Decay the Learning Rate,
  Increase the Batch Size", ICLR 2018, arXiv:1711.00489.
- P. Goyal, P. Dollár, R. Girshick, P. Noordhuis, L. Wesolowski, A. Kyrola,
  A. Tulloch, Y. Jia, K. He, "Accurate, Large Minibatch SGD: Training ImageNet in 1
  Hour", arXiv:1706.02677.
- Jastrzębski, Kenton, Arpit, Ballas, Fischer, Bengio, Storkey,
  "Three Factors Influencing Minima in SGD", ICANN 2018, arXiv:1711.04623.
- S. Mandt, M. D. Hoffman, D. M. Blei, "Stochastic Gradient Descent as Approximate
  Bayesian Inference", JMLR 18, 2017, arXiv:1704.04289.
- C. J. Shallue, J. Lee, J. Antognini, J. Sohl-Dickstein, R. Frostig, G. E. Dahl,
  "Measuring the Effects of Data Parallelism on Neural Network Training", JMLR 20,
  2019, arXiv:1811.03600.
- G. Zhang, L. Li, Z. Nado, J. Martens, S. Sachdeva, G. E. Dahl, C. J. Shallue,
  R. Grosse, "Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a
  Noisy Quadratic Model", NeurIPS 2019, arXiv:1907.04164.
- S. Ma, R. Bassily, M. Belkin, "The Power of Interpolation: Understanding the
  Effectiveness of SGD in Modern Over-parametrized Learning", arXiv:1712.06559.
- Golmant, Vemuri, Yao, Feinberg, Gholami, Rothauge, Mahoney, Gonzalez, "On the
  Computational Inefficiency of Large Batch Sizes for Stochastic Gradient Descent",
  arXiv:1811.12941.
- J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray,
  A. Radford, J. Wu, D. Amodei, "Scaling Laws for Neural Language Models",
  arXiv:2001.08361.
- H. Zhang, D. Morwani, N. Vyas, J. Wu, D. Zou, U. Ghai, D. Foster, S. Kakade, "How
  Does Critical Batch Size Scale in Pre-training?", ICLR 2025, arXiv:2410.21676.
- W. Merrill, S. Arora, D. Groeneveld, H. Hajishirzi, "Critical Batch Size Revisited:
  A Simple Empirical Approach to Large-Batch Language Model Training", NeurIPS 2025,
  arXiv:2505.23971.
- Bergsma, Dey, Gosal, Gray, Soboleva, Hestness, "Power Lines:
  Scaling Laws for Weight Decay and Batch Size in LLM Pre-training", NeurIPS 2025,
  arXiv:2505.13738.
- S. Li and co-authors, "Surge Phenomenon in Optimal Learning Rate and Batch Size
  Scaling", arXiv:2405.14578.
- Marek, Lotfi, Somasundaram, Wilson, Goldblum, "Small Batch Size
  Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation
  Is Wasteful", arXiv:2507.07101.
- N. Sato, H. Naganuma, H. Iiduka, "Convergence Bound and Critical Batch Size of Muon
  Optimizer", arXiv:2507.01598.
- Naganuma, Gupta, Briki, Mitliagkas, Rish, Raman, Shi, "Adaptive Batch
  Sizes Using Non-Euclidean Gradient Noise Scales for Stochastic Sign and Spectral
  Descent", arXiv:2602.03001.

## Muon and its relatives
- K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, J. Bernstein, "Muon: An
  optimizer for hidden layers in neural networks", blog post, 8 December 2024
  (kellerjordan.github.io/posts/muon), with the repositories KellerJordan/Muon and
  KellerJordan/modded-nanogpt.
- J. Bernstein, L. Newhouse, "Old Optimizer, New Norm: An Anthology",
  arXiv:2409.20325.
- J. Bernstein, L. Newhouse, "Modular Duality in Deep Learning", arXiv:2410.21265.
- J. Bernstein, "Deriving Muon", blog post, 7 March 2025
  (jeremybernste.in/writing/deriving-muon).
- J. Liu, J. Su, X. Yao and co-authors (Moonshot AI), "Muon is Scalable for LLM
  Training", arXiv:2502.16982.
- V. Gupta, T. Koren, Y. Singer, "Shampoo: Preconditioned Stochastic Tensor
  Optimization", ICML 2018, arXiv:1802.09568.
- N. Vyas, D. Morwani, R. Zhao, M. Kwun, I. Shapira, D. Brandfonbrener, L. Janson,
  S. Kakade, "SOAP: Improving and Stabilizing Shampoo using Adam", arXiv:2409.11321,
  with the authors' code (github.com/nikhilvyas/SOAP).
- D. Carlson, V. Cevher, L. Carin, "Stochastic Spectral Descent for Restricted
  Boltzmann Machines", AISTATS 2015, PMLR 38:111-119.
- M. Tuddenham, A. Prügel-Bennett, J. Hare, "Orthogonalising gradients to speed up
  neural network optimisation", arXiv:2202.07052.
- T. Pethick, W. Xie, K. Antonakopoulos, Z. Zhu, A. Silveti-Falls, V. Cevher,
  "Training Deep Learning Models with Norm-Constrained LMOs" (Scion),
  arXiv:2502.07529.
- K. Ahn, B. Xu, N. Abreu, Y. Fan, G. Magakyan, P. Sharma, Z. Zhan, J. Langford,
  "Dion: Distributed Orthonormalized Updates", arXiv:2504.05295.
- C. Si, D. Zhang, W. Shen, "AdaMuon: Adaptive Muon Optimizer", arXiv:2507.11005.
- T. T.-K. Lau, Q. Long, W. Su, "PolarGrad: A Class of Matrix-Gradient Optimizers from
  a Unifying Preconditioning Perspective", arXiv:2505.21799.
- N. Amsel, D. Persson, C. Musco, R. M. Gower, "The Polar Express: Optimal Matrix Sign
  Methods and Their Application to the Muon Algorithm", arXiv:2505.16932.
- Z. Li, L. Liu, C. Liang, W. Chen, T. Zhao, "NorMuon: Making Muon more efficient and
  scalable", arXiv:2510.05491.
- Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv:2507.20534.
- L. Chen, J. Li, Q. Liu, "Muon Optimizes Under Spectral Norm Constraints",
  arXiv:2506.15054.
- S. Xie, Z. Li, "Implicit Bias of AdamW: ℓ∞ Norm Constrained Optimization",
  arXiv:2404.04454.
- D. Kovalev, "Understanding Gradient Orthogonalization for Deep Learning via
  Non-Euclidean Trust-Region Optimization", arXiv:2503.12645.
- Li, Hong, "A Note on the Convergence of Muon", arXiv:2502.02900 (abstract only).
- Shen and co-authors, "On the Convergence Analysis of Muon", arXiv:2505.23737
  (abstract only).

## Muon against Adam: measurements and explanations
- Essential AI (I. Shah, A. M. Polloreno, K. Stratos and co-authors), "Practical
  Efficiency of Muon for Pretraining", arXiv:2505.02222.
- K. Wen, D. Hall, T. Ma, P. Liang, "Fantastic Pretraining Optimizers and Where to
  Find Them", arXiv:2509.02046.
- A. Semenov, M. Pagliardini, M. Jaggi, "Benchmarking Optimizers for Large Language
  Model Pretraining", arXiv:2509.01440.
- X. Wu, W. Yu, C. Zhang, P. Woodland, "Spectral Allocation: Why Muon Outperforms
  Adam, and How to Improve Muon", arXiv:2608.25990.
- S. Wang, F. Zhang, J. Li, C. Du, C. Du, T. Pang, Z. Yang, M. Hong, V. Y. F. Tan,
  "Muon Outperforms Adam in Tail-End Associative Memory Learning", arXiv:2509.26030.
- K. Frans, P. Abbeel, S. Levine, "What Really Matters in Matrix-Whitening
  Optimizers?", arXiv:2510.25000.
- C. Fan, M. Schmidt, C. Thrampoulidis, "Implicit Bias of Spectral Descent and Muon on
  Multiclass Separable Data", arXiv:2502.04664.
- D. Davis, D. Drusvyatskiy, "When do spectral gradient updates help in deep
  learning?", arXiv:2512.04299.
- Khona and co-authors, "SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales",
  arXiv:2607.20548 (abstract only).
- Wu, Mansouri, Zhu, "From Spectral to Frobenius: When Muon Loses Its Spectral
  Implicit Bias", arXiv:2610.20112 (abstract only).
- Beneventano, Abdelmoneum, Poggio, "The Spectral Dynamics and Noise Geometry of
  Muon", arXiv:2606.08388 (abstract only).
- Qi and co-authors, "Delving into Muon and Beyond: Deep Analysis and Extensions",
  arXiv:2602.04669 (abstract only).
- Dolatabadi and co-authors, "NuMuon: Nuclear-Norm-Constrained Muon for Compressible
  LLM Training", arXiv:2603.03597 (abstract only).

## The edge of stability
- J. M. Cohen, S. Kaur, Y. Li, J. Z. Kolter, A. Talwalkar, "Gradient Descent on Neural
  Networks Typically Occurs at the Edge of Stability", ICLR 2021, arXiv:2103.00065.
- J. M. Cohen, B. Ghorbani, S. Krishnan, N. Agarwal, S. Medapati, M. Badura, D. Suo,
  D. Cardoze, Z. Nado, G. E. Dahl, J. Gilmer, "Adaptive Gradient Methods at the Edge
  of Stability", arXiv:2207.14484.
- A. Damian, E. Nichani, J. D. Lee, "Self-Stabilization: The Implicit Bias of Gradient
  Descent at the Edge of Stability", ICLR 2023, arXiv:2209.15594.
- A. Lewkowycz, Y. Bahri, E. Dyer, J. Sohl-Dickstein, G. Gur-Ari, "The large learning
  rate phase of deep learning: the catapult mechanism", arXiv:2003.02218.
- X. Zhu, Z. Wang, X. Wang, M. Zhou, R. Ge, "Understanding Edge-of-Stability Training
  Dynamics with a Minimalist Example", arXiv:2210.03294.
- A. Agarwala, F. Pedregosa, J. Pennington, "Second-order regression models exhibit
  progressive sharpening to the edge of stability", arXiv:2210.04860.
- S. Arora, Z. Li, A. Panigrahi, "Understanding Gradient Descent on Edge of Stability
  in Deep Learning", ICML 2022, arXiv:2205.09745.
- J. M. Cohen, A. Damian, A. Talwalkar, J. Z. Kolter, J. D. Lee, "Understanding
  Optimization in Deep Learning with Central Flows", ICLR 2025, arXiv:2410.24206.
- L. Wu, C. Ma, W. E, "How SGD Selects the Global Minima in Over-parameterized
  Learning: A Dynamical Stability Perspective", NeurIPS 2018.
- S. Jastrzębski, M. Szymczak, S. Fort, D. Arpit, J. Tabor, K. Cho, K. Geras, "The
  Break-Even Point on Optimization Trajectories of Deep Neural Networks", ICLR 2020,
  arXiv:2002.09572.
- A. Andreyev, P. Beneventano, "Edge of Stochastic Stability: Revisiting the Edge of
  Stability for SGD", arXiv:2412.20553.
- S. Lee, C. Jang, "A New Characterization of the Edge of Stability Based on a
  Sharpness Measure Aware of Batch Gradient Distribution", ICLR 2023 (abstract only).
- F. Liao, A. Kolomvaki, A. Kyrillidis, "SGD at the Edge of Stability: The Stochastic
  Sharpness Gap", arXiv:2604.21016.

## Warm-up and schedules
- J. Gilmer, B. Ghorbani, A. Garg, S. Kudugunta, B. Neyshabur, D. Cardoze, G. Dahl,
  Z. Nado, O. Firat, "A Loss Curvature Perspective on Training Instability in Deep
  Learning", arXiv:2110.04369.
- D. S. Kalra, M. Barkeshli, "Why Warmup the Learning Rate? Underlying Mechanisms and
  Improvements", NeurIPS 2024, arXiv:2406.09405.
- I. Loshchilov, F. Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts",
  ICLR 2017, arXiv:1608.03983.
- K. Wen, Z. Li, J. Wang, D. Hall, P. Liang, T. Ma, "Understanding Warmup-Stable-Decay
  Learning Rates: A River Valley Loss Landscape Perspective", arXiv:2410.05192
  (abstract only).

## The Hessian of a network
- L. Sagun, U. Evci, V. U. Guney, Y. Dauphin, L. Bottou, "Empirical Analysis of the
  Hessian of Over-Parametrized Neural Networks", arXiv:1706.04454.
- B. Ghorbani, S. Krishnan, Y. Xiao, "An Investigation into Neural Net Optimization
  via Hessian Eigenvalue Density", arXiv:1901.10159.
- V. Papyan, "The Full Spectrum of Deepnet Hessians at Scale: Dynamics with SGD
  Training and Sample Size", arXiv:1811.07062.
- V. Papyan, "Measurements of Three-Level Hierarchical Structure in the Outliers in
  the Spectrum of Deepnet Hessians", arXiv:1901.08244.
- V. Papyan, "Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra",
  arXiv:2008.11865.
- G. Gur-Ari, D. A. Roberts, E. Dyer, "Gradient Descent Happens in a Tiny Subspace",
  arXiv:1812.04754.
- M. Song, K. Ahn, C. Yun, "Does SGD really happen in tiny subspaces?", ICLR 2025,
  arXiv:2405.16002.

## Weight decay, normalisation and the effective learning rate
- T. van Laarhoven, "L2 Regularization versus Batch and Weight Normalization",
  arXiv:1706.05350.
- E. Hoffer, R. Banner, I. Golan, D. Soudry, "Norm matters: efficient and accurate
  normalization schemes in deep networks", arXiv:1803.01814.
- G. Zhang, C. Wang, B. Xu, R. Grosse, "Three Mechanisms of Weight Decay
  Regularization", arXiv:1810.12281.
- Z. Li, S. Arora, "An Exponential Learning Rate Schedule for Deep Learning",
  arXiv:1910.07454.
- R. Wan, Z. Zhu, X. Zhang, J. Sun, "Spherical Motion Dynamics: Learning Dynamics of
  Neural Network with Normalization, Weight Decay, and SGD", arXiv:2006.08419.
- A. Kosson, B. Messmer, M. Jaggi, "Rotational Equilibrium: How Weight Decay Balances
  Learning Across Neural Networks", ICML 2024, arXiv:2305.17212.
- F. D'Angelo, M. Andriushchenko, A. Varre, N. Flammarion, "Why Do We Need Weight
  Decay in Modern Deep Learning?", arXiv:2310.04415.

## Sharpness and generalisation
- P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur, "Sharpness-Aware Minimization for
  Efficiently Improving Generalization", arXiv:2010.01412.
- L. Dinh, R. Pascanu, S. Bengio, Y. Bengio, "Sharp Minima Can Generalize For Deep
  Nets", arXiv:1703.04933.
- M. Andriushchenko, F. Croce, M. Müller, M. Hein, N. Flammarion, "A Modern Look at
  the Relationship between Sharpness and Generalization", ICML 2023,
  arXiv:2302.07011.
