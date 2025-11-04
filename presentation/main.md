---
theme: uncover
paginate: skip
---

<style>
section {
  font-size: 30px;
  background-color:white;
}
h1 {
  font-size: 2rem;
}
.columns {
    display: grid;
    grid-template-columns: repeat(2, auto);
    align-items: center;
    gap: 1rem;
}
</style>

<div class="columns">
<div>

# CrispyMcMark
- Elia Gatti
- Pietro Ventrucci
- Filippo Marcon

</div>
<div>

![width:700](./mcmark.png)

</div>
</div>


---

# How we reached the final implementation
- found papers about DWT-SVD
- began with DWT-SVD on the whole image
- then transitioned to blocks, embedding two singular values for each block
- switched to embedding one singular value per block

---


<!-- paginate: true -->

# Embedding
- reshape watermark to 32x32 and take $U_w,V_w,W_w = SVD(watermark)$
- $U_w,W_w$ hardcoded in the detection, $V_w$ are the singular values
- choose $x$ square blocks
- forall $i<x$:
    - take $LL_b$ of the DWT of $blocks[i]$ 
    - compute the singular values $V_b$ of its $LL$
    - embed $V[i]$ into the first singular value($V_b[0]$) 
    - inverse the first two steps to reconstruct the block, and put it back into the image 

---

![width:900px](./embedding.png)

---

# Detection
- use watermarked - original image to find $x$ watermarked blocks
- attacked watermark = difference between singular values of attacked blocks and original blocks  
- original watermark = difference between singular values of watermarked blocks and original blocks  
- detection compares the two extracted watermarks using threshold


---

![width:1100px](./extraction.png)

---

# Implementation challenges 
- Embedding quality performed better on images with high entropy zones
- On low entropy images embedding was more visible 
- Not enough time to refine the design and try different techniques
- Multiplicative embedding was harder to tweak
- Understanding how the algorithm performed/finding bugs based on the ROC function. 

---

# ROC1: Original 
![width:600px](./roc_original.png)
<br>

---

<!-- nella presentazione spiegare che qui è uguale ma che quando stavamo facendo sviluppo veniva diverso/ci ha forzato a cambiare threshold -->
# ROC2
![width:600px](./roc_3.png)
ROC1 + label 0 for original(attacked) images + label 0 for destroyed 

---

# Effects of hardcoding the watermark 
<!-- if we try to embed a different watermark than our group's the algorithm is shit -->
![width:600px](./roc_broken.png)
<br>

---

# Possible Improvements 
- Add redundancy based on singular value importance 
- Improve the invisibility of attack squares, either by block choice or embedding strength
- explore different embedding techniques on each block

--- 


# Attack Strategy
- binary search to find optimal attack strength
- attack functions tweaked to accept parameter $0\leq\beta\leq1$
- use of masks to attack different areas of the image
- parallelization to improve execution speed

---


# Attack Strategy - binary search
![width:1000px](./attack_plot_new.png)

---

# Attack Strategy - masks
![width:1000px](./burger_masks.png)

---

# Attack Strategy - GUI tool
![width:700px](./gui.png)

---

<div class="columns">
<div>

# Questions?

</div>
<div>

![width:500](./gorilla.jpg)

</div>
</div>
