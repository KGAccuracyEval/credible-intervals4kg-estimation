import numpy as np


class RandomErrorModel(object):
    """
    The Random Error Model (REM) labels the triples in the KG using a fixed error rate in [0, 1]
    """

    def annotateKG(self, kg, errorP):
        """
        Annotate the KG based on the REM fixed error rate

        :param kg: target KG
        :param errorP: fixed error rate
        :return: computed ground truth for the KG
        """

        # generate labels where prob(0) = errorP and prob(1) = 1-errorP
        labels = np.random.choice([0, 1], size=len(kg), p=[errorP, 1-errorP])

        # associate labels w/ triple IDs to create the ground truth
        groundTruth = {kg[i][0]: labels[i] for i in range(len(kg))}
        return groundTruth
