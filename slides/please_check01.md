# Please check: Lecture 1 (Introduction to Machine Learning)

Items in `lecture01.tex` that need a check by the lecturer before the slides are final.

## Facts and sources
- [ ] **Citizen reports case study** (`lecture01.tex:544`): "more than 170,000 reports per year" comes from Rudinac's 2025 slides, where the slide shows 173,940 per year without a source. The frame cites "Rudinac, 2025 lecture slides". The frame describes only the setup of Sukel, Rudinac and Worring (2019), not their results.
- [ ] **COMPAS frame** (`lecture01.tex:647`): the one-sentence summary of Angwin et al. (2016) was written from memory of the article. Check that the wording ("Black defendants who did not reoffend were more often labeled high risk than white defendants") matches the article.
- [ ] **References added from memory**: Mitchell (1997), Breiman (2001), Sculley et al. (2015), Pedregosa et al. (2011) and Silver et al. (2016) are standard works, but their bibliographic details were not checked against the sources in this session. Blei (2012), the ProPublica article and the IEEE Spectrum link come from last year's slides.

## Adapted examples
- [ ] **Closing sale** and **two schools** examples: paraphrased from Müller's 2017 lecture notes and credited as "Müller, 2017".
- [ ] **Music service proxy** (`lecture01.tex:615`): adapted from Müller's Spotify example. The "worse proxy: clicks" example is new and not from Müller.

## Course logistics
- [ ] **Background frame** (`lecture01.tex:177`, note at `:187`): "high-school mathematics and a first statistics course" and "the labs build [programming] up" describe Diadié Sow's labs; confirm with him.
- [ ] **Setup frame**: points to Diadié Sow's setup guide, which does not exist yet. Update the frame once his guide is available (name of the document, where students find it, Anaconda or Colab).

## Template
- [ ] Every frame logs a 34 pt overfull `\hbox`. It comes from the theme (`sibbeamer2026.sty`): the unmodified `template.tex` produces it too.
