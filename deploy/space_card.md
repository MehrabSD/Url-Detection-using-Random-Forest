---
title: Malicious URL Detector
emoji: 🔗
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Tree ensembles that flag malicious URLs from text alone
---

# Malicious URL detector

Type a web address and a random forest averaged with a gradient-boosted model decides
whether it looks benign or malicious
(phishing, malware delivery, defacement) **from the text of the address alone**. The link is
never opened. The page then explains the verdict: which stretches of the address look
suspicious, what each of the two models said, how the forest's 200 trees voted, and how far
each of the 56 features moved the
score.

It also **attacks itself**: *Try to sneak it past the model* searches for the cheapest realistic
edits that get a flagged link through, against the model as trained or a hardened multi-view
scorer (one free edit evaded the plain model on 83.1 % of caught phishing URLs, the hardened one
on 12.7 %). A lookalike check spells out typosquats and homoglyphs letter by letter, *Check a
whole message* ranks every link in a pasted email, and QR codes can be read from a photo or the
camera (decoded in your browser; the image is never uploaded).

All figures on the page are read from the saved model and its evaluation reports. They were
measured on websites that were kept out of training entirely, so they describe how the model
does on sites it has never seen.

- Trained on 1.65 million URLs from three public datasets (Kaggle Malicious URLs, PhiUSIIL,
  a site-aware URL benchmark) and checked on a fourth (Hannousse & Yahiouche 2021).
- scikit-learn random forest + histogram gradient boosting (600 rounds), served by a Python standard-library HTTP server; the page is
  plain HTML, CSS and JavaScript with no build step.

This Space was uploaded by `deploy/push_to_hf_space.py` from the project folder. The project's
`TECHNICAL_DOCUMENTATION.md` explains the data, features, model, results and limitations.
