from numpy import ones, newaxis, max

from .SFLDT import SFLDT

class FuzzySFLDT(SFLDT):
    """
    The FuzzySFLDT diagnoser.
    
    This diagnoser is an extension of the SFLDT diagnoser, where the participation of the nodes is fuzzy.
    """
    
    def update_fuzzy_participation(self,
     ) -> None:
        """
        Update the fuzzy participation of the nodes.
        The participation is calculated as the depth of the component normalized by the depth of the path.
        """
        super().update_fuzzy_participation()
        if len(self.spectra.shape) > len(self.components_depths_vector.shape):
            self.components_depths_vector = self.components_depths_vector[:, newaxis]
        self.spectra = self.spectra * self.components_depths_vector / self.paths_depths_vector[newaxis, :]
        assert max(self.spectra) <= 1.0, f"Participation should be in [0, 1] but got {max(self.spectra)}"