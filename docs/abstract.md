# Abstract

We developed a reproducible pipeline for de novo antimicrobial peptide design that combines physicochemically conditioned language-model generation with model-based prioritization of antimicrobial potency and hemolytic safety.

We first analyzed the organizer-provided reference set of 39,448 antimicrobial peptides against a length-matched background of putative non-AMP sequences from UniProt. Physicochemical descriptors were compared to identify interpretable properties associated with antimicrobial peptides. Net charge at pH 7.4 and Wimley–White interfacial hydrophobicity at pH 8 were selected as complementary conditioning variables. Their joint distributions were modeled using two-dimensional kernel density estimation, and five representative target conditions were selected from regions enriched in known AMPs.

ProtGPT3-1.3B was first adapted to the organizer AMP dataset using LoRA and then extended with continuous soft-prompt conditioning on the two selected physicochemical properties. Equal sampling at the five target conditions produced a pool of 250,000 unique candidate peptides after removal of invalid sequences, duplicates, and exact matches to the organizer reference set.

Candidates were prioritized using two independently trained predictors. Antimicrobial activity was scored by a LightGBM ranking model trained on GRAMPA MIC measurements using ESM-2 embeddings and physicochemical descriptors. Hemolytic safety was estimated as HC50 using an ESM-2-based censored regression model trained on erythrocyte hemolysis measurements. Candidates were ranked by equally weighted ranks of predicted antimicrobial activity and predicted HC50. Sequences with normalized Levenshtein similarity above 0.80 to any organizer reference peptide were subsequently excluded.

The 50,000 highest-ranked remaining peptides constitute the submitted library, and its first 100 sequences form the ranked Top100 set. The complete generation and selection pipeline uses a fixed software environment and random seed and reproduces the submitted FASTA files deterministically on the tested hardware.
