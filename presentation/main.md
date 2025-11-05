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
<!-- 


After some research on the subject we decide to stick with DWT-SVD it and embed the whole watermark in the LL subband of the whole image, this had two main problem, low wPSNR and also low robustness. 

We tried to improve the robustness by using multiplicative approach without good result, so we decided to stick with additive. 

Later, We thought about a block based approach, embedding two singular values in block 4x4 this yielded far better robustness but made the first block of the watermarked image more visible. 

To improve visibility we decided to: firstly, embed a singular value per block, secondly, made them a bit larger (8x8) to make it blend more with the image and, lastly embed them in high entropy zones.

And this basically is really close to the current solution for the embedding...
-->
# How we reached the final implementation
- initial DWT-SVD implementation
- multiplicative embedding
- initial block implementation
- final block implementation

---



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
<!-- 
The current approach start by taking the original image and selecting the best blocks scored on overall entropy (The higher the better).

We also compute the SVD of the watermark extracting the singular values

After we compute the DWT and SVD of each block extracting the singular values. We compute the additive embedding for each block with only one singular value of the watermark embedded always in the first position.

Lastly we compute the inverse SVD and inverse DWT reconstructing the overall watermark image.

The U and V component of the watermark SVD result are hardcoded in the detection (as per challnge rules).

This technique allowed us in general to emebed less information compared to the 1024 bits. Making us able to embed with a high wPSNR and high robustness.

The choice of using entropy made hiding the watermarked block easier but it did not always work.
 -->
![width:900px](./embedding.png)

---

# Detection
- use watermarked - original image to find $x$ watermarked blocks
- attacked watermark = difference between singular values of attacked blocks and original blocks  
- original watermark = difference between singular values of watermarked blocks and original blocks  
- detection compares the two extracted watermarks using threshold


---
<!-- 
For the detection we exploited the fact that we had access to also the watermarked image, finding the block locations as simple as doing a diff between watermarked and original collecting all of the blocks. 

This was possible becasue we embed them in the same order as we retrieve them, so that we are always sure that the block that we retrieve first is the first block.

After retrieving the block location we compute DWT-SVD on both the original and watermark, this gave us the singular values of the watermark which after inverse of SVD (using the original hardcoded U and V component) gave us the original watermark.

The same process was applied for the difference using the original and attacked image. 

After we recovered both watermark we compute the Bit Error Rate similarity which had a Thresold computed using the ROC of 0.7 

 -->
![width:1100px](./extraction.png)

---

<!-- So, we built our watermarking tool, but it has its limits.

Quickly, our embedding works better on complex images with lots of detail, but it is more visible on simple images.
We also ran out of time to try other techniques, and just understanding what our results meant was not so straighforward.

This brings me to the ROC curve, which is the scorecard we used to see how well our technique worked -->

# Current Limitations
- Embedding quality performed better on images with high entropy zones
- On low entropy images embedding was more visible 
- Not enough time to refine the design and try different techniques
- Understanding how the algorithm performed/finding bugs based on the ROC function. 

---

<!-- This was our first score. An AUC of 1.0. A perfect square.

It looked a little too perfect... and it was.

We discovered our watermark was found also in non watermarked images, and destroyed images.

That perfect score wasn't real. -->

# ROC
![width:600px](./roc_original.png)
<br>

---

<!-- To fix this, we had to create a proper 'H0' set. We fed the detector images that were truly destroyed, and we also added original, non-watermarked images. We essentially told the ROC, 'You are not allowed to find a watermark in these.'

This is the result. A much more honest curve. An AUC of 0.980, a score we can actually trust. This realistic curve is what allowed us to find the correct, balanced threshold for detection -->

# ROC
![width:600px](./roc_3.png)
ROC1 + label 0 for original(attacked) images + label 0 for destroyed 

---

# Effects of hardcoding the watermark 
<!-- But this whole process revealed a... funny quirk.

Our detector is, let's say, very loyal. To make it work so well, we had to hardcode parts of our specific watermark, the U and W vectors, right into the detection function. It's trained to find one watermark only.

So, what happens if you try to find a different watermark?

Well, you get this. An AUC of 0.398.

At this point, you would have a more accurate detection function... by just flipping a coin.

So, the takeaway? Our algorithm is fantastic... as long as you're not trying to find any watermark but its own! Thank you -->

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

---

# Don't go on

---

# Trust me, the presentation is over

---

# Special thanks

<div class="columns">
  <div>
  
  - **Cursed lama** - for making our nights less lonely
  - **Claudio** - for parallelization
  - **Our opponents** - for a fair "challenge"

  </div>

  <div>

  ![width:500](./cursed_lama.jpg)

  </div>
</div>