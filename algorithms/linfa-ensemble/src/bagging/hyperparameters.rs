// TODO: Make this actually bagging. Use spec provided
use crate::decision_trees::DecisionTreeParams;
use linfa::{
    error::{Error, Result},
    Float, Label,
};

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

/// The set of hyperparameters that can be specified for fitting a
/// [extra trees](struct.ExtraTrees.html).
///
/// ### Example
///
/// ```rust
/// use linfa_trees::{ExtraTrees, DecisionTree, SplitQuality};
/// use linfa_datasets::iris;
/// use linfa::prelude::*;
///
/// // Initialize the default set of parameters
/// let decision_tree_params = DecisionTree::params();
/// let params = ExtraTrees::params();
/// // Set the parameters to the desired values
/// let decision_tree_params = decision_tree_params.split_quality(SplitQuality::Entropy).max_depth(Some(5)).min_weight_leaf(2.).random_split(true);
/// let params = params.decision_tree_params(decision_tree_params);
///
///
/// // Load the data
/// let (train, val) = linfa_datasets::iris().split_with_ratio(0.9);
/// // Fit the decision tree on the training data
/// let trees = params.fit(&train).unwrap();
/// // Predict on validation and check accuracy
/// let val_accuracy = trees.predict(&val).confusion_matrix(&val).unwrap().accuracy();
/// assert!(val_accuracy > 0.99);
/// ```
///
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug)]
pub struct ExtraTreesParams<F, L> {
    pub decision_tree_params: DecisionTreeParams<F, L>,
    pub num_estimators: usize,
}

impl<F: Float, L: Label> ExtraTreesParams<F, L> {
    /// Sets the hyperparameters for each inidividual decision tree
    pub fn decision_tree_params(mut self, decision_tree_params: DecisionTreeParams<F, L>) -> Self {
        self.decision_tree_params = decision_tree_params;
        self
    }

    /// Sets the total number of trees that are generated
    pub fn num_estimators(mut self, num_estimators: usize) -> Self {
        self.num_estimators = num_estimators;
        self
    }

    /// Checks the correctness of the hyperparameters
    ///
    /// ### Panics
    ///
    /// If the minimum impurity increase is not greater than zero
    pub fn validate(&self) -> Result<()> {
        if !self.decision_tree_params.random_split {
            return Err(Error::Parameters(format!(
                "Extra trees should always use a random split, but random_split is currently set to {}",
                false.to_string()
            )));
        }

        Ok(())
    }
}
