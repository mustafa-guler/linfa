// TODO: Change anything related to decision trees
use linfa::{error::Result, traits::*, Float, Label};

use ndarray::{Array1, Array2};

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
pub struct BaggingParams<O> {
    pub num_estimators: usize,
    pub max_n_rows: Option<usize>,
    pub estimator_params: O,
}

/// Indirection helper trait used to specify the generic subtypes of the ensemble members.
pub trait Indirection {
    type F: Float;
    type L: Label;
}

impl<O> BaggingParams<O>
where
    O: Indirection + Fit<Array2<O::F>, Array1<O::L>, linfa::Error>,
    O::Object: PredictInplace<Array2<O::F>, Array1<O::L>>,
{
    /// Sets the max size of the bootstrapped dataset. Use `None` for whatever the dataset size is
    pub fn max_n_rows(mut self, max_n_rows: Option<usize>) -> Self {
        self.max_n_rows = max_n_rows;
        self
    }

    /// Sets the number of ensemble members that are generated
    pub fn num_estimators(mut self, num_estimators: usize) -> Self {
        self.num_estimators = num_estimators;
        self
    }

    /// Checks the correctness of the hyperparameters
    ///
    /// ### Panics
    ///
    /// Not yet implemented
    pub fn validate(&self) -> Result<()> {
        // TODO: Is there anything that we need to check?

        Ok(())
    }
}
