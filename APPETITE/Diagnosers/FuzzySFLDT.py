from numpy import ones, newaxis

from .SFLDT import SFLDT

class FuzzySFLDT(SFLDT):
    def update_fuzzy_participation(self,
     ) -> None:
        self.spectra = self.spectra * (self.components_depths_vector + ones(self.node_count))[:, newaxis] / self.paths_depths_vector[newaxis, :]