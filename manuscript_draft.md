> **Formatting note for Google Docs:** Open a new Google Doc → click Edit → select "Paste from Markdown" → paste this file. All headings, bold text, italics, and tables will render automatically.

---

# MOSAIC: an accessible desktop pipeline for automated classification of dyadic social behavior from pose-tracking data

**Authors:** Abdullah Jawwad Yousafi¹, [★ FILL IN: Evan Andrews full name]², [★ FILL IN: Emily Dew full name]², [★ FILL IN: other contributors], Qin Wang²\*

**Affiliations:**
¹ Department of Computer Science and Mathematics, Beloit College, Beloit, WI, USA
² Department of Neuroscience and Regenerative Medicine, Medical College of Georgia, Augusta University, Augusta, GA, USA

\*Corresponding author: Qin Wang ([★ FILL IN: email])

---

## Abstract

Quantifying social behavior in freely interacting rodents is central to modeling the social deficits seen in neurodevelopmental and psychiatric conditions, yet it remains bottlenecked by manual frame-by-frame scoring, which is slow, subjective, and difficult to reproduce across observers and laboratories. Deep-learning pose estimation tools such as SLEAP and DeepLabCut have automated the tracking of animal keypoints, but their raw output is noisy and describes only where body parts are, not what animals are doing. Existing behavior-classification packages either require substantial programming and machine-learning setup or return abstract, unnamed movement motifs that are hard to map onto an ethogram. We present MOSAIC, a desktop application that converts multi-animal SLEAP output into cleaned trajectories, an interpretable panel of per-frame kinematic and social features, segmented behavior bouts, and classified social behaviors, without requiring the user to write code. MOSAIC applies a four-stage denoising pipeline (monotonic interpolation, Kalman smoothing, and median plus Savitzky-Golay filtering), engineers features organized around three non-overlapping body zones (head, body, tail), segments continuous behavior using gap-bridging (0.25 s) and minimum-duration (0.8 s) rules, and classifies 11 distinct second-order social behaviors with a random forest evaluated under animal-grouped cross-validation. On a dyadic free-interaction dataset, MOSAIC reached [★ FILL IN: accuracy]% cross-validated accuracy across 11 social behavior categories, and its automated proximity and contact calls agreed strongly with blinded manual scoring (proximity: 91.9% accuracy, Cohen's kappa 0.836; contact: 87.2% accuracy, kappa 0.720 across 298 scored frames). MOSAIC lowers the technical barrier to reproducible social-behavior phenotyping and is designed to slot directly onto an existing SLEAP workflow.

**Keywords:** social behavior; pose estimation; SLEAP; behavior classification; random forest; open-source software; computational neuroethology; mouse

---

## 1. Introduction

Social behavior tests are a workhorse of preclinical neuroscience because impaired social interaction is a core, cross-diagnostic feature of conditions including autism spectrum disorder and schizophrenia [1,2]. Freely moving dyadic paradigms, in which two animals interact in an open arena, capture a far richer behavioral repertoire than fixed apparatus assays, including approach, following, sniffing directed at different body regions, and physical contact [3]. That richness is also their central problem: extracting quantitative measures from these interactions has traditionally required a trained observer to code behavior frame by frame. Manual scoring is labor intensive, is prone to observer drift and inter-rater disagreement, and scales poorly, which directly limits the throughput and reproducibility of exactly the subtle social phenotypes that disease models are built to detect [4].

The first half of this problem has been substantially solved. Markerless pose-estimation systems, notably DeepLabCut [5] and SLEAP [6], use convolutional neural networks to track user-defined keypoints on one or more animals from ordinary video, producing per-frame coordinates for each body part. SLEAP in particular is designed for multi-animal tracking and is well suited to dyadic social recordings. These tools have transformed the front end of behavioral analysis, but they deliberately stop at tracking. Their output is a stream of keypoint coordinates that is both noisy — with missing values where tracking drops out, identity swaps, and frame-to-frame jitter — and semantically empty, in that it never states whether the two animals are sniffing, following, or ignoring one another. The gap between raw pose coordinates and labeled, quantified behavior is precisely where analysis effort now concentrates [7].

A number of tools address this second stage. Among supervised classifiers, SimBA [8] takes human-annotated examples of target behaviors and trains machine-learning models on features computed from pose data. JAABA [9], an earlier supervised system, introduced the combination of frame-level annotation with random-forest classification and has been widely applied to *Drosophila* and rodent datasets. More recently, LabGym [10] introduced a video-clip-based deep-learning classifier that does not require explicit keypoint tracking. Unsupervised approaches, including B-SOiD [11], VAME [12], and Keypoint-MoSeq [13], instead discover recurring movement motifs directly from pose dynamics without any labels, which reduces annotation burden and can surface behavioral structure a human might not pre-specify. Each family carries a cost. Supervised pipelines demand a nontrivial investment in annotation, feature configuration, and model tuning, and much of that workflow still assumes comfort with scripting or with configuring a technical environment. Unsupervised pipelines remove the labeling burden but return clusters or syllables that are abstract and must be interpreted and mapped back onto named, ethologically meaningful behaviors after the fact, which is its own expert task.

For a laboratory whose members are neuroscientists rather than software engineers, neither option is turnkey. What is missing is an accessible tool that goes from raw multi-animal SLEAP output directly to interpretable, named, quantified social behaviors, through a transparent feature layer a biologist can reason about, inside an interface that requires no programming. We built MOSAIC to fill that gap. MOSAIC is a PySide6/Qt desktop application that ingests SLEAP tracking exports and, without any code from the user, reconstructs clean kinematic trajectories, computes an interpretable panel of per-frame features spanning locomotion, posture, spatial use, and social geometry, segments continuous behavior into bouts, and classifies social behaviors with a supervised random forest. It exposes every intermediate stage for inspection and exports per-animal and pairwise summaries as tables and figures ready for downstream statistics.

This paper describes MOSAIC's design and implementation, and validates it in two ways: by cross-validated classification accuracy across a large behavioral repertoire, and by direct agreement between MOSAIC's automated proximity and contact calls and blinded manual scoring. We developed and tested MOSAIC in the context of an ongoing study of Chd7 function in the anterior cingulate cortex and its role in social behavior [14,15], a setting in which reliable, high-throughput behavioral phenotyping is the rate-limiting step.

---

## 2. Materials and Methods

### 2.1 Design overview and architecture

MOSAIC is implemented in Python 3.14 as a desktop application using the PySide6 6.11 binding to the Qt framework, chosen for responsive cross-platform rendering, native file handling, and stable embedding of Matplotlib canvases. An earlier prototype built on Tkinter was prone to interface freezes and crashes on lower-specification laboratory machines, primarily because heavy computation ran on the interface thread; MOSAIC moves all expensive work (loading, denoising, feature extraction, classification) onto Qt worker threads so the interface stays responsive throughout.

The pipeline is organized as a linear sequence of stages, each of which writes an inspectable intermediate representation: (i) data ingestion from SLEAP output, (ii) trajectory denoising, (iii) per-frame feature extraction, (iv) bout segmentation, and (v) behavior classification, followed by (vi) export. Users interact with this pipeline through a graphical interface that also supports arena region-of-interest definition, frame-level inspection of tracking and feature values, and a real-time interaction display.

[★ FILL IN: Figure 1 — horizontal pipeline schematic showing raw HDF5 → denoising → features → bouts → classifier → export]

### 2.2 Input data and skeleton definition

MOSAIC consumes the HDF5 analysis exports produced by SLEAP for two-animal recordings. The standard SLEAP analysis export is an HDF5 file containing at minimum three datasets: **tracks**, a floating-point array of per-frame, per-node x and y coordinates for each animal; **node_names**, a list of skeleton node labels; and **track_names**, a list of track (animal identity) labels. MOSAIC reads all three, validates that array dimensions are internally consistent, and constructs a frame-index map that correctly handles both dense exports (where every video frame is present) and sparse exports (where SLEAP has tracked only a subset of frames, recorded in a separate **frame_idx** dataset). When a sparse frame_idx is absent, MOSAIC falls back to sequential integer indices. Edge connectivity (the **edge_inds** dataset) is read if present and used only for visualization; it is not required for any analysis. Tracks are stored internally in the shape (n_frames, 2, n_nodes, n_animals), where the second axis carries x (index 0) and y (index 1) coordinates in pixel units.

The skeleton used in the present study comprises eight nodes per animal: nose, left ear (ear_l), right ear (ear_r), body center (body), left hip (hip_l), right hip (hip_r), tail base (tail_base), and tail end (tail_end). The tail_end node is excluded from all proximity and contact computations because it is the least reliably tracked point and, in practice, contributes phantom near-contact events when it is poorly localized.

[★ FILL IN: Recording and tracking setup — mouse strain/genotype and sex, age at testing, number of animals and dyads, arena dimensions (cm), camera model, frame rate (fps), resolution, session duration, and lighting. Also report the SLEAP model: backbone architecture, number of labeled frames, train/validation split, and a tracking accuracy figure such as mean keypoint localization error in pixels or millimeters and mAP. Include your IACUC/animal protocol approval number and approving institution.]

### 2.3 Trajectory denoising

Raw multi-animal pose estimates contain gaps, jitter, and outliers that corrupt any velocity- or geometry-based feature computed from them. MOSAIC applies a fixed four-stage denoising pipeline to each node coordinate time series before feature extraction.

**Gap filling by monotonic interpolation.** Short runs of missing values (gaps ≤ 0.25 s at the recording frame rate, corresponding to ≤ 6 frames at 24 fps) are filled using piecewise cubic Hermite interpolating polynomial (PCHIP) interpolation [16], which preserves monotonicity between tracked points and avoids the overshoot that cubic splines introduce around abrupt movements. The 0.25 s limit was chosen to reconstruct transient occlusions without fabricating long-duration trajectories.

**Kalman smoothing.** Gaps longer than 0.25 s that survive PCHIP filling are reconstructed using a Kalman filter [17] operating on a local context window of 1.5 s around the gap. The filter uses a constant-velocity motion model with transition matrix [[1, 1], [0, 1]] and observation matrix [[1, 0]], transition covariance 10⁻³ × I₂ and observation covariance 10⁻² × I₁ (identity-scaled). Only the positions at missing frames are written back; surrounding observed positions are not modified.

**Median filtering.** A sliding median filter with a window of 3 frames removes residual single-frame spikes and outliers that survive interpolation.

**Savitzky-Golay filtering.** A Savitzky-Golay filter [18] with a window of 5 frames and polynomial order 2 provides final low-pass smoothing that preserves the shape and peak timing of movement events. A separate, wider Savitzky-Golay pass with a minimum window of 9 frames and polynomial order 3 is applied specifically to the body-axis heading computed from landmark pairs, because arctan2 of position differences amplifies per-pixel jitter more than direct position smoothing does.

We deliberately did not mask low-confidence nodes outright; masking created discontinuities that were more damaging to downstream geometry than smoothing-based reconstruction. All denoising parameters (PCHIP gap threshold, median window, and Savitzky-Golay window and polynomial order) are fixed at the values above and are not exposed to the user, to ensure consistent behavior across sessions.

### 2.4 Body-zone definitions

Contact and proximity between two animals are only meaningful when defined between specific body regions, since a nose-to-nose event and a nose-to-tail event are ethologically distinct. MOSAIC groups nodes into three non-overlapping body zones per animal: **head** (nose, ear_l, ear_r), **body** (body center, hip_l, hip_r), and **tail** (tail_base only). All zone-referenced proximity and contact metrics are computed as the minimum inter-node distance between the relevant zone on one animal and the relevant zone on the other, rather than centroid-to-centroid only, so that a close approach at any point in a zone is registered. This zone scheme replaced an earlier centroid-only definition that failed to fire even when nose or hip nodes were within a centimeter of each other.

### 2.5 Feature extraction

From the denoised tracks MOSAIC computes per-frame features organized into four interpretable groups (see Supplementary Table S1 for the complete list):

**Locomotion.** Per-node speed, x- and y-velocity components, acceleration magnitude, and jerk (rate of change of acceleration), all computed via Savitzky-Golay differentiation (window 3, polynomial order 3) applied to the smoothed coordinate time series, scaled by the recording frame rate. Immobility, walking, and running states are derived from body-center speed thresholds (stationary < 3 cm s⁻¹; walking 3–20 cm s⁻¹; running > 20 cm s⁻¹). A turning flag is raised when the absolute angular velocity of the body axis exceeds 30 deg s⁻¹, and a direction-reversal flag is raised when a sign reversal in angular velocity occurs within ±2 frames on both sides, with both sides exceeding 20 deg s⁻¹. First-order state labels are smoothed with a median filter of width max(3, round(0.1 × fps)) frames to remove isolated single-frame state flickers.

**Pose geometry.** Body-axis heading is derived from the rear-to-front anatomical landmark pair (body center → nose), with a fallback chain to hip midpoint → ear midpoint, hip midpoint → nose, and finally velocity heading. Heading is represented in the sin-cos domain before smoothing to avoid circular wrap-around errors at 0°/360°, then converted back to degrees in the range [−180, 180]. Angular velocity of the body axis and an hourglass-area ratio (the ratio of anterior to posterior body-section triangle areas formed by head, body, and hip nodes) capture postural shape.

**Spatial use.** Occupancy of the user-defined arena region of interest and of nine sub-zones — four corners (C1–C4), four wall strips (W1–W4), and open center — is recorded per frame. Sub-zones are defined by a strip of width equal to one-eighth of the shorter arena dimension measured from each wall. Path efficiency is computed as the ratio of straight-line displacement to cumulative path length over a sliding window.

**Social and pairwise geometry.** Inter-animal center-of-mass distance and all six zone-pair distances (head-head, head-body, head-tail, body-body, body-tail, tail-tail), all reported in centimeters after scaling by a user-supplied pixels-per-centimeter factor. General proximity (any zone pair ≤ 3.0 cm) and contact (any zone pair ≤ 1.0 cm) are binary flags derived from these distances. Relative orientation (the angle from animal A's heading to the vector connecting the two body centers) and approach velocity (the time derivative of inter-animal distance) capture dynamic social geometry.

All spatial features are computed in the video's native pixel coordinate space and then converted to centimeters. Features restricted to the arena region of interest use a coordinate transform that maps drawn ROI coordinates through Qt's item-scene-view hierarchy back into native video pixel coordinates, ensuring that tracking data and ROI boundaries are aligned.

### 2.6 Bout segmentation

Frame-level social signals are noisy and, if read frame by frame, fragment a single continuous behavior into many spurious short events. MOSAIC converts frame-level proximity and contact signals into discrete behavior bouts using two rules, both applied in frame units at the recording frame rate:

**Gap bridging** merges two events separated by a gap of at most max(1, round(0.25 × fps)) frames (6 frames at 24 fps, approximately 0.25 s) into a single bout. **Minimum bout duration** discards bouts shorter than max(1, round(0.8 × fps)) frames (approximately 0.8 s) as likely tracking noise. These thresholds were chosen empirically: the 0.25 s gap limit bridges single-step tracking dropouts, and the 0.8 s minimum suppresses spurious sub-second events that our manual validation showed were predominantly false positives. For each behavior MOSAIC reports bout count, mean bout duration, and total time.

### 2.7 Behavior classification

On top of the feature and bout layers, MOSAIC classifies social behavior using a random forest [19], implemented with scikit-learn 1.8.0 [20].

**Behaviors.** Behavior is described in two tiers. First-order locomotor states (stationary, walking, running, turning, direction-reversal) are defined by the speed and angular-velocity thresholds described in §2.5 and are applied per animal. Second-order social behaviors are pairwise and directional where applicable. The pipeline classifies 11 distinct second-order behaviors, represented as 19 binary flags: Follow (A→B, B→A), Chase (A→B, B→A), Flee (A→B, B→A), Approach (A→B, B→A), Active Avoidance (A→B, B→A), Passive Avoidance (A→B, B→A), Stationary Proximity (symmetric), Social Orientation (A→B, B→A), Disengaged (symmetric), Visual Attention (A→B, B→A), and Auditory Attention (A→B, B→A). Behavior definitions are detailed in Supplementary Table S1.

**Training data.** [★ FILL IN: This section is critical for peer review. You must describe: (1) how many recording sessions were used for training; (2) how many total bouts were annotated; (3) who performed the annotation and with what tool; (4) the per-class sample counts or the class imbalance range; (5) how annotation frames relate to the 298-frame manual validation set — ideally the validation frames are completely held out from training, and if so, state this explicitly.]

**Classifier.** A random forest with 100 decision trees is trained on bout-level features. Bout-level features are computed as aggregated summaries (mean, standard deviation, and 95th percentile) of the per-frame feature values within each bout. Class imbalance is addressed by balanced class weighting, which adjusts sample weights inversely proportional to class frequency.

**Cross-validation.** When the number of unique animal-pair groups is at least 5, cross-validation uses GroupKFold with 5 folds, assigning all bouts from the same pair to the same fold. This prevents the classifier from achieving inflated accuracy by memorizing individual animals rather than learning the behavior. When fewer than 5 unique groups are present (small datasets), StratifiedKFold is used as a fallback. Classes with fewer than 5 bouts are excluded from evaluation. All reported accuracy figures are out-of-fold predictions from cross-validation, not training-set performance.

**Feature importance.** Feature importance is assessed by permutation importance (5 repetitions) computed on the full final model, rather than impurity-based (Gini) importance, because the engineered features are continuous and correlated — a regime in which impurity-based importance is known to be biased toward high-cardinality features.

### 2.8 Graphical interface and user workflow

MOSAIC's interface is built so a researcher with no programming experience can run the full pipeline. The user loads a video and its paired SLEAP HDF5 file, then draws the arena region of interest directly on the video frame. Because the preview is displayed at a scaled resolution, MOSAIC maps the drawn region back into the video's native pixel coordinates using an explicit item-scene-view coordinate transform, so the region aligns exactly with the SLEAP tracks; getting this transform wrong silently corrupts every spatial feature computed inside the region, so it is handled carefully and tested against known frame dimensions.

The interface exposes each intermediate stage. Users can step through any frame and inspect both the overlaid tracking and the computed feature values for that frame, which makes the pipeline auditable rather than a black box. A real-time display panel reports playback position, per-animal speed and current sub-zone (using the nine-zone C1–C4/W1–W4/Open system), an interaction-state indicator (apart, proximity, or contact) read from the same segmented bouts used in the summary, and running behavioral tallies up to the current frame. On completion, MOSAIC exports per-animal and pairwise summary tables and figures in spreadsheet and PDF formats for downstream statistical analysis.

### 2.9 Validation against manual scoring

To assess whether MOSAIC's automated calls match a human observer, a blinded human scorer coded proximity and contact on [★ FILL IN: describe the sample — which session(s), which dyad(s), frame-sampling method (random stratified / systematic / consecutive), scoring software used, and whether the scorer was blinded to animal genotype and MOSAIC output] across 298 frames, and these labels were compared frame by frame against MOSAIC's automated output.

[★ FILL IN: Inter-rater reliability — a second blinded scorer should independently code the same 298 frames. Report the inter-human kappa as the ceiling against which MOSAIC's kappa should be interpreted. Without this ceiling, reviewers cannot assess whether the residual disagreement between MOSAIC and the human scorer exceeds human-human disagreement.]

Agreement between MOSAIC and the human scorer was quantified using overall accuracy and Cohen's kappa [21], which corrects for chance agreement. Bootstrapped 95% confidence intervals on kappa were computed by resampling the 298 frames with replacement 10,000 times. [★ FILL IN: confirm this was done, or compute and insert confidence intervals.] Confusion matrices were generated for both proximity and contact, and receiver operating characteristic (ROC) analysis was used to assess sensitivity-specificity trade-offs across threshold values.

The 298 manually scored frames are [★ FILL IN: confirm: completely held out from any training data used for the classifier reported in §3.2, or describe how they relate to training data].

### 2.10 Implementation and availability

MOSAIC is written in Python 3.14 using PySide6 6.11, NumPy, SciPy, pandas, scikit-learn 1.8.0, pykalman, OpenCV, and Matplotlib. It runs on Windows 10/11 (tested; [★ FILL IN: add macOS/Linux if verified]). Source code, documentation, and a worked example are available at [★ FILL IN: repository URL, e.g., https://github.com/WangLabs/MOSAIC] under the [★ FILL IN: license, e.g., MIT] license, archived at [★ FILL IN: Zenodo DOI]. [★ FILL IN: Confirm repository and release plan with Dr. Wang before submitting.]

---

## 3. Results

### 3.1 Denoising recovers usable trajectories from noisy pose output

The four-stage pipeline filled short tracking gaps, removed single-frame outliers, and produced smooth derivatives suitable for velocity, acceleration, and jerk features that were otherwise dominated by noise in the raw tracks. Across [★ FILL IN: n sessions], [★ FILL IN: mean ± SD]% of frames per node contained missing values before denoising. Of these, [★ FILL IN: %] were short enough (≤ 0.25 s) to be filled by PCHIP interpolation; the remainder were reconstructed by Kalman smoothing.

[★ FILL IN: Support with Figure 2 — before/after trajectory panels or a per-node missing-data summary (% frames with NaN per node, per session). Consider also showing jitter reduction quantitatively: e.g., per-node standard deviation of position during periods of verified immobility, before vs. after smoothing.]

### 3.2 The classifier separates social behaviors under grouped cross-validation

Trained on labeled bouts and evaluated with animal-grouped (GroupKFold, k = 5) cross-validation, the random forest achieved [★ FILL IN: accuracy]% overall cross-validated accuracy across 11 second-order social behaviors. [★ FILL IN: Report per-class recall range, and note which behaviors were most reliably classified and which were most confused. For example: "Per-class recall ranged from X% to Y%; the most reliable classes were [A, B, C], while [D and E] were most often confused with each other, consistent with their overlapping kinematic signatures." Support with Figure 3: confusion matrix heatmap.]

Permutation importance identified [★ FILL IN: top 3–5 features from your actual permutation importance output, e.g., "inter-animal distance, head-head zone distance, and relative orientation angle"] as the most informative features for social behaviors, consistent with the ethological expectation that proximity and relative orientation carry most of the social signal. [★ FILL IN: Support with Figure 4: top-20 global permutation importance bar chart from the rf_analysis.pdf output.]

### 3.3 Automated calls agree with blinded manual scoring

Across 298 manually scored frames, MOSAIC's automated proximity classification reached 91.9% accuracy (95% CI [★ FILL IN: e.g., 88.1–95.2%]) with a Cohen's kappa of 0.836 (95% CI [★ FILL IN: e.g., 0.771–0.893]), and its contact classification reached 87.2% accuracy (95% CI [★ FILL IN: e.g., 83.2–91.0%]) with a kappa of 0.720 (95% CI [★ FILL IN: e.g., 0.636–0.800]). Under standard interpretation, kappa values above 0.8 indicate near-perfect agreement and values from 0.6 to 0.8 indicate substantial agreement [21], so MOSAIC's proximity calls are essentially interchangeable with the human scorer and its contact calls agree substantially. The residual disagreement was concentrated in [★ FILL IN: describe where errors occurred — e.g., "brief ambiguous approach-to-contact transitions lasting fewer than three frames, where the minimum-bout rule causes MOSAIC to withhold a contact call that the human scorer assigns"].

[★ FILL IN: Support with Figure 5: side-by-side confusion matrices for proximity and contact, and Figure 6: rolling frame-by-frame agreement timeline showing where MOSAIC and the human scorer diverge.]

[★ FILL IN: If two human scorers were used, report the inter-human kappa here and compare directly: "MOSAIC's proximity kappa (0.836) was within the range of inter-human agreement (kappa = X.XXX), indicating that automated scoring was at human-level reliability."]

### 3.4 Throughput and practical efficiency

[★ FILL IN: This section requires real measurements before submission. Measure and report: (1) how long manual coding of one full recording session requires in person-hours; (2) how long MOSAIC takes to process the same session end-to-end on stated hardware (CPU, RAM, operating system); (3) what the resulting time saving is. Example: "A [length]-minute recording session that required approximately [X] person-hours to score manually was processed by MOSAIC in [Y] minutes on a [CPU] machine with [RAM] GB RAM, representing a [Z]-fold reduction in analysis time."]

---

## 4. Discussion

MOSAIC occupies a specific and, we argue, underserved position in the behavior-analysis landscape. Relative to supervised toolkits such as SimBA [8] and JAABA [9], MOSAIC shares the core strategy of computing pose-derived features and training a random forest, but it is packaged as a guided desktop workflow with an explicit, biologically interpretable feature layer and built-in frame-level auditing, lowering the setup and scripting burden for laboratories without dedicated computational staff. Relative to video-based classifiers such as LabGym [10], MOSAIC produces explainable features that can be inspected frame by frame, rather than a learned embedding. Relative to unsupervised motif-discovery methods such as B-SOiD, VAME, and Keypoint-MoSeq [11,12,13], MOSAIC returns named, ethologically defined behaviors rather than abstract clusters, which is what a phenotyping study comparing genotypes or treatments ultimately needs, at the cost of requiring the target behaviors to be defined in advance. MOSAIC is therefore best understood not as a replacement for these tools but as an accessible bridge from SLEAP output to interpretable social phenotypes.

Several design decisions were driven by correctness problems that are easy to get wrong and quietly corrupt results. Defining proximity and contact over three non-overlapping body zones, rather than centroid-to-centroid, was necessary to register close approaches that a centroid measure misses. Excluding the poorly tracked tail-end node eliminated phantom contact events. Using circular statistics for all angular features (heading, angular velocity) avoided wrap-around errors at 0° and 360°. Grouping cross-validation by animal identity prevents the inflated accuracy that individual-level data leakage would produce. We highlight these because they are the kinds of decisions that determine whether an automated pipeline is trustworthy, and they are often invisible in a finished tool.

The main limitations are honest and bounded. The manual-scoring validation, while showing strong agreement, was conducted on a modest 298-frame sample and on proximity and contact specifically; broader validation across the full behavioral repertoire, across multiple scorers to establish an inter-rater ceiling, and across additional cohorts and genotypes is the natural next step. The classifier's accuracy figure reflects performance on the present dataset and should be re-established when the tool is applied to new recording conditions or different skeleton configurations. MOSAIC currently targets two-animal interactions with a fixed skeleton and would need extension to support larger groups or different skeletons. Finally, because target behaviors are defined in advance, MOSAIC will not surface behaviors that were not anticipated, which is exactly the strength of the unsupervised methods and suggests a productive pairing: unsupervised discovery to define an ethogram, MOSAIC to score it reproducibly at scale.

In the context of the Chd7/anterior cingulate cortex work motivating this tool [14,15], MOSAIC directly addresses the rate-limiting step. Establishing whether a genetic perturbation shifts subtle social behavior requires scoring many hours of interaction consistently, and it is precisely subtle phenotypes that human scoring is most likely to miss or apply inconsistently. By making that scoring automated, reproducible, and auditable, MOSAIC turns behavioral phenotyping from a bottleneck into a scalable readout.

---

## 5. Conclusion

MOSAIC provides an accessible, end-to-end path from noisy multi-animal SLEAP pose output to cleaned trajectories, interpretable features, segmented bouts, and classified social behaviors, inside a desktop application that requires no programming. It reached [★ FILL IN: accuracy]% cross-validated accuracy across 11 second-order social behavior categories and agreed strongly with blinded manual scoring on proximity and contact (proximity kappa 0.836, contact kappa 0.720). By combining the interpretability of supervised, feature-based classification with a guided interface built for biologists, MOSAIC makes reproducible social-behavior phenotyping practical for laboratories that lack in-house computational infrastructure.

---

## Declarations

**Code availability:** [★ FILL IN: repository URL, license, Zenodo DOI.]

**Data availability:** [★ FILL IN: statement about where validation frames and labels are deposited.]

**Funding:** STAR Research Fellowship, Medical College of Georgia; [★ FILL IN: any grant numbers supporting the Wang Lab work.]

**Author contributions:** [★ FILL IN: CRediT-style — e.g., A.J.Y. designed and implemented the software, performed the analysis, and drafted the manuscript; co-authors contributed biological data, experimental design, and critical revision; Q.W. supervised the project; all authors approved the final manuscript.]

**Competing interests:** The authors declare no competing interests.

**Ethics:** All animal procedures were approved by the [★ FILL IN: Augusta University IACUC], protocol [★ FILL IN: number].

**Acknowledgments:** We thank the Wang Lab, [★ FILL IN: Emily Dew] for the biological framework, and the STAR Research Fellowship Program.

---

## References

1. American Psychiatric Association. *Diagnostic and Statistical Manual of Mental Disorders*, 5th ed. (2013).
2. Insel TR. Rethinking schizophrenia. *Nature* 468, 187–193 (2010).
3. Silverman JL, Yang M, Lord C, Crawley JN. Behavioural phenotyping assays for mouse models of autism. *Nat Rev Neurosci* 11, 490–502 (2010).
4. Anderson DJ, Perona P. Toward a science of computational ethology. *Neuron* 84, 18–31 (2014).
5. Mathis A, et al. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. *Nat Neurosci* 21, 1281–1289 (2018).
6. Pereira TD, et al. SLEAP: a deep learning system for multi-animal pose tracking. *Nat Methods* 19, 486–495 (2022).
7. Pereira TD, Shaevitz JW, Murthy M. Quantifying behavior to understand the brain. *Nat Neurosci* 23, 1537–1549 (2020).
8. Nilsson SR, et al. Simple Behavioral Analysis (SimBA) — an open source toolkit for computer classification of complex social behaviors in experimental animals. [★ FILL IN: verify full citation for the 2024 Nat Neurosci version]
9. Kabra M, Robie AA, Rivera-Alba M, Branson S, Branson K. JAABA: interactive machine learning for automatic annotation of animal behavior. *Nat Methods* 10, 64–67 (2013).
10. Hu H, et al. LabGym: quantification of user-defined animal behaviors using learning-based holistic assessment. *Cell Rep Methods* 3, 100415 (2023).
11. Hsu AI, Yttri EA. B-SOiD, an open-source unsupervised algorithm for identification of spontaneous behaviors. *Nat Commun* 12, 5188 (2021).
12. Luxem K, et al. Identifying behavioral structure from deep variational embeddings of animal motion. *Commun Biol* 5, 684 (2022).
13. Weinreb C, et al. Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics. *Nat Methods* 21, 1329–1339 (2024).
14. [★ FILL IN: CHD7/ACC reference — Emily Dew's in-preparation manuscript or an appropriate published CHD7 reference.]
15. [★ FILL IN: second CHD7/ACC reference if needed.]
16. Fritsch FN, Carlson RE. Monotone piecewise cubic interpolation. *SIAM J Numer Anal* 17, 238–246 (1980).
17. Kalman RE. A new approach to linear filtering and prediction problems. *J Basic Eng* 82, 35–45 (1960).
18. Savitzky A, Golay MJE. Smoothing and differentiation of data by simplified least squares procedures. *Anal Chem* 36, 1627–1639 (1964).
19. Breiman L. Random forests. *Mach Learn* 45, 5–32 (2001).
20. Pedregosa F, et al. Scikit-learn: machine learning in Python. *J Mach Learn Res* 12, 2825–2830 (2011).
21. Cohen J. A coefficient of agreement for nominal scales. *Educ Psychol Meas* 20, 37–46 (1960).

---

## Supplementary Table S1 — Complete feature list

All features computed per frame by MOSAIC. Features marked **(A)** are per-animal; features marked **(P)** are pairwise (computed for each dyad). Speed and distance features are converted to cm s⁻¹ and cm respectively using the user-supplied pixel-to-centimeter scale.

### Per-animal kinematic features (A)

Computed independently for each tracked node (nose, ear_l, ear_r, body, hip_l, hip_r, tail_base):

| Feature | Units | Description |
|---|---|---|
| node_x | px | x-coordinate after denoising |
| node_y | px | y-coordinate after denoising |
| node_speed | cm s⁻¹ | Euclidean speed from Savitzky-Golay differentiation |
| node_vx | cm s⁻¹ | x-velocity component |
| node_vy | cm s⁻¹ | y-velocity component |
| node_accel | cm s⁻² | Acceleration magnitude |
| node_jerk | cm s⁻³ | Jerk (rate of change of acceleration) |
| node_total_disp | px | Cumulative displacement from session start |

### Per-animal body-level features (A)

| Feature | Units | Description |
|---|---|---|
| body_heading_deg | deg | Body-axis heading, body→nose landmark pair, circular-mean smoothed |
| angular_velocity | deg s⁻¹ | First derivative of body-axis heading |
| hourglass_area | px² | Area of anterior body-section triangle (nose, ear_l, ear_r) |
| hourglass_ratio | — | Anterior triangle area / posterior triangle area (hip_l, hip_r, body) |
| curvature | deg s⁻¹ | Absolute angular velocity; proxy for turning tightness |
| speed_accel | cm s⁻² | Rate of change of body-center speed |
| path_efficiency | — | Displacement / cumulative path length; 1.0 = straight line |
| dist_roi_center | px | Distance from body center to arena ROI centroid |
| dist_roi_boundary | px | Distance from body center to nearest ROI boundary |
| position_entropy | bits | Shannon entropy of spatial position (arena subdivided into grid) |

### First-order locomotor state flags (A)

All state flags smoothed with median filter width max(3, round(0.1 × fps)).

| Feature | Type | Definition |
|---|---|---|
| stationary | binary | body-center speed < 3 cm s⁻¹ |
| walking | binary | 3 ≤ body-center speed ≤ 20 cm s⁻¹ |
| running | binary | body-center speed > 20 cm s⁻¹ |
| turning | binary | absolute body-axis angular velocity > 30 deg s⁻¹ |
| dir_reversal | binary | Sign reversal of angular velocity within ±2 frames, both sides > 20 deg s⁻¹ |

### Zone occupancy features (A)

| Feature | Type | Description |
|---|---|---|
| zone | categorical | C1, C2, C3, C4 (corners), W1, W2, W3, W4 (walls), or Open (centre) |

Corners defined as within strip width of both adjacent walls; wall zones as within strip width of one wall only. Strip width = 1/8 × shorter arena dimension.

### Pairwise distance and proximity features (P)

Computed for each pair of animals in the session.

| Feature | Units | Description |
|---|---|---|
| inter_animal_dist | cm | Center-of-mass distance between animals |
| head_head_dist | cm | Minimum inter-node distance between head zones of both animals |
| head_body_dist_AtoB | cm | Min distance: animal A head zone → animal B body zone |
| head_body_dist_BtoA | cm | Min distance: animal B head zone → animal A body zone |
| head_tail_dist_AtoB | cm | Min distance: animal A head → animal B tail zone |
| head_tail_dist_BtoA | cm | Min distance: animal B head → animal A tail zone |
| body_body_dist | cm | Min distance: body zones of both animals |
| body_tail_dist_AtoB | cm | Min distance: animal A body → animal B tail |
| body_tail_dist_BtoA | cm | Min distance: animal B body → animal A tail |
| tail_tail_dist | cm | Min distance: tail zones of both animals |
| within_3cm | binary | Any zone-pair distance ≤ 3.0 cm (general proximity flag) |
| within_1cm | binary | Any zone-pair distance ≤ 1.0 cm (general contact flag) |
| relative_orientation_AtoB | deg | Angle from A's heading to the A→B inter-animal vector |
| relative_orientation_BtoA | deg | Angle from B's heading to the B→A inter-animal vector |
| approach_velocity | cm s⁻¹ | Rate of change of inter-animal distance; negative = approaching |

### First-order contact detection flags (P)

Derived from zone-pair distances; applied to segmented bouts before second-order classification.

| Behavior | Definition |
|---|---|
| NoseNose | Head-head zone distance ≤ contact threshold (1.0 cm) |
| NoseHead A→B | Animal A nose zone ≤ contact threshold of animal B head zone |
| NoseHead B→A | Animal B nose zone ≤ contact threshold of animal A head zone |
| NoseBody A→B | Animal A nose zone ≤ contact threshold of animal B body zone |
| NoseBody B→A | Animal B nose zone ≤ contact threshold of animal A body zone |
| NoseRear A→B | Animal A nose zone ≤ contact threshold of animal B tail zone |
| NoseRear B→A | Animal B nose zone ≤ contact threshold of animal A tail zone |

### Second-order pairwise behavior flags (P)

All binary (1 = active, 0 = inactive), after gap bridging and minimum-bout filtering. A→B and B→A variants are defined symmetrically; only the A→B definition is given.

| Behavior | Direction | Definition |
|---|---|---|
| Follow | A→B | Animal A locomoting in the same direction as and within proximity of animal B; A's heading aligns with B's velocity vector (face error < 50°) |
| Chase | A→B | Animal A running toward animal B while B is fleeing; A accelerating, face error < 45°, B's path efficiency high and directed away |
| Flee | A→B | Animal A running away from approaching animal B; A's face error toward B > 130°, high jerk, A retreating |
| Approach | A→B | Animal A moving toward animal B from non-proximate position; face error < 55°, positive approach velocity, path efficiency elevated |
| Active Avoidance | A→B | Animal A actively moving away when animal B approaches; face error > 120°, A retreating while B's approach velocity is negative |
| Passive Avoidance | A→B | Animal A stationary while animal B approaches; A does not redirect toward B |
| Stationary Proximity | symmetric | Both animals stationary and within general proximity threshold |
| Social Orientation | A→B | Animal A facing animal B (face error < 45° when stationary, < 65° when active) within proximity distance, not fleeing or actively avoiding |
| Disengaged | symmetric | Both animals outside proximity threshold with no recent social contact |
| Visual Attention | A→B | Animal B is within animal A's visual field: face error (angle from A's heading to direction-toward-B) < 120° |
| Auditory Attention | A→B | Animal B is within animal A's auditory field: face error < 150° (broader lateral range than visual attention) |

**Face error** is defined as the absolute angle in [0°, 180°] between the subject animal's body-axis heading vector and the bearing from the subject to the target animal.

---

## ★ Remaining items — must be completed before submission

The following items cannot be extracted from code and require experimental data or decisions from the research team:

| # | Item | Location |
|---|---|---|
| 1 | Real classification accuracy | Abstract, §3.2, Conclusion |
| 2 | Recording setup: strain, sex, age, n dyads, arena size, camera, fps, resolution, session length | §2.2 |
| 3 | SLEAP model: backbone, n labeled frames, train/val split, mAP or keypoint error | §2.2 |
| 4 | IACUC protocol number and approving institution | §2.2, Ethics |
| 5 | Training data: n sessions, n bouts, annotator(s), annotation tool, class balance | §2.7 |
| 6 | Whether validation frames (298) are held out from training — state explicitly | §2.9 |
| 7 | Validation sampling method (random/stratified/consecutive) and scorer details | §2.9 |
| 8 | Inter-rater reliability: second human scorer on same 298 frames | §2.9, §3.3 |
| 9 | 95% confidence intervals on all reported kappa/accuracy values | §3.3 |
| 10 | Description of where classification errors occur (which behaviors confused) | §3.2 |
| 11 | Throughput timing: manual scoring time vs. MOSAIC processing time | §3.4 |
| 12 | Figure 2: denoising before/after comparison | §3.1 |
| 13 | Figure 3: confusion matrix from classification results | §3.2 |
| 14 | Figure 4: top-20 permutation importance bar chart (from rf_analysis.pdf) | §3.2 |
| 15 | Figure 5: confusion matrices for proximity and contact validation | §3.3 |
| 16 | Repository URL and Zenodo DOI | §2.10, Declarations |
| 17 | License choice | §2.10 |
| 18 | Author full names and corresponding email | Title page |
| 19 | CHD7/ACC references (refs 14–15) | References |
| 20 | SimBA 2024 Nat Neurosci full citation (ref 8) | References |

**Items now filled from code (no longer need user input):**
- Python 3.14, PySide6 6.11 (§2.10)
- Second-order behavior count and full list (§2.7, Table S1)
- AuditoryAttn and VisualAttn definitions with exact thresholds (Table S1)
- SocialOrient definition with exact thresholds (Table S1)
- First-order contact flags table (Table S1)
- Face error definition (Table S1)
- Abstract behavior count (now reads "11 distinct second-order social behaviors")
- SLEAP HDF5 format technical description (§2.2)

---

*MOSAIC manuscript draft — [★ FILL IN: target journal] submission.*
*Codebase version: MOSIAC 2.0 — generated 2026-07-09.*
