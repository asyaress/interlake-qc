# Release checklist (user actions)

Do these with your own GitHub/Zenodo accounts. This repository does not create tags or log into services.

1. Review `CITATION.cff` and `.zenodo.json` (author order, ORCID, affiliations).
2. On Zenodo: enable the GitHub integration for `asyaress/interlake-qc`.
3. On GitHub: create Release **v1.0.0** (tag `v1.0.0`) from the reviewed commit.
4. Wait for Zenodo to mint a DOI for the release.
5. Copy the DOI into:
   - README “How to cite”
   - `CITATION.cff` (`doi:` field)
6. Optionally push a small docs-only commit with the DOI filled in.
7. Confirm the Zenodo landing page shows MIT license and the four creators in the correct order.
