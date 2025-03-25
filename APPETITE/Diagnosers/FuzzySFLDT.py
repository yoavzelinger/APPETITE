from numpy import ones, newaxis

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
        self.spectra = self.spectra * (self.components_depths_vector + ones(self.node_count))[:, newaxis] / self.paths_depths_vector[newaxis, :]