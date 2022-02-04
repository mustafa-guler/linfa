use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2};

use super::hyperparameters::*;
use crate::decision_trees::*;
use linfa::{
    dataset::{AsTargets, Labels},
    error::Error,
    error::Result,
    traits::*,
    DatasetBase, Float, Label,
};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde_crate::{de::DeserializeOwned, Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(
        crate = "serde_crate",
        bound = "L: DeserializeOwned + Serialize, F: DeserializeOwned + Serialize"
    )
)]
#[derive(Debug)]
pub struct ExtraTrees<F: Float, L: Label> {
    all_trees: Vec<TreeNode<F, L>>,
    num_features: usize,
}

impl<F: Float, L: Label + std::fmt::Debug> ExtraTrees<F, L> {
    /// Defaults are provided if the optional parameters are not specified:
    /// * `decision_tree_params = Default parameters for [decision trees](struct.DecisionTreeParams.html) with random_split overriden`
    /// * `num_estimators = 100`
    /// * `max_features = None`
    /// The `max_features` default of `None` will be overriden to the square root of the number of features when a dataset is provided.
    pub fn params() -> ExtraTreesParams<F, L> {
        ExtraTreesParams {
            decision_tree_params: DecisionTree::params().random_split(true).max_features(None),
            num_estimators: 100,
        }
    }

    pub fn predict_ranked(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array1<Vec<L>> {
        let mut y: Array1<Vec<L>> = Array1::from_elem(x.nrows(), vec![]);
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        for (row, target) in x.rows().into_iter().zip(y.iter_mut()) {
            *target = make_ranked_prediction(self, &row);
        }

        y
    }
}

impl<'a, F: Float, L: Label + 'a + std::fmt::Debug, D, T> Fit<ArrayBase<D, Ix2>, T, Error>
    for ExtraTreesParams<F, L>
where
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>,
{
    type Object = ExtraTrees<F, L>;

    /// Fit extremely randomized trees using hyperparameters in `self` on the `dataset`
    /// consisting of a matrix of features and an array of labels.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let num_features = dataset.records().ncols();
        self.decision_tree_params.validate(num_features)?;
        self.validate()?;

        let (records, feature_names) = (dataset.records(), dataset.feature_names());

        // Overrides the `max_features` hyperparameter once the dataset is provided
        // to the square root of the number of features if it is `None`
        let true_max_features = self
            .decision_tree_params
            .max_features
            .unwrap_or_else(|| (num_features as f64).sqrt() as usize);

        let mut all_trees = Vec::with_capacity(self.num_estimators);

        // Create all decision trees
        for _ in 0..self.num_estimators {
            let setup = set_up_dataset(records, &feature_names);

            let mut root_node = TreeNode::fit(
                dataset,
                &setup.0,
                &self.decision_tree_params,
                &setup.1,
                0,
                Some(true_max_features),
            )?;

            root_node.prune();
            all_trees.push(root_node);
        }

        Ok(ExtraTrees {
            all_trees,
            num_features,
        })
    }
}

impl<F: Float, L: Label + Default, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<L>>
    for ExtraTrees<F, L>
{
    /// Make predictions for each row of a matrix of features `x`.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        for (row, target) in x.rows().into_iter().zip(y.iter_mut()) {
            *target = make_prediction(self, &row);
        }
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}

/// Classify a sample &x
fn make_prediction<F: Float, L: Label>(
    model: &ExtraTrees<F, L>,
    x: &ArrayBase<impl Data<Elem = F>, Ix1>,
) -> L {
    use crate::decision_trees::algorithm::make_prediction;

    // Count predictions across all trees
    let mut prediction_frequencies: HashMap<L, f32> = HashMap::with_capacity(model.num_features);
    for root_node in &model.all_trees {
        let prediction = make_prediction(x, root_node);
        let value = prediction_frequencies.entry(prediction).or_default();
        *value += 1.0;
    }

    // Return most frequent prediction
    find_modal_class(&prediction_frequencies)
}

/// Obtain the probability distribution over labels for an input row of the data.
fn get_prediction_freqs<F: Float, L: Label>(
    model: &ExtraTrees<F, L>,
    x: &ArrayBase<impl Data<Elem = F>, Ix1>,
) -> HashMap<L, f32> {
    let mut prediction_frequencies: HashMap<L, f32> = HashMap::with_capacity(model.num_features);

    // Verify that prediction probs is frequency, and not already weighted by the number of samples
    for root_node in &model.all_trees {
        let (prediction_proba, num_samples) = make_prediction_distribution(x, root_node);
        let num_samples = num_samples as f32;
        for (key, value) in &*prediction_proba {
            let freq = prediction_frequencies.entry(key.clone()).or_insert(0.0f32);
            *freq += value * num_samples;
        }
    }

    prediction_frequencies
}

// /// For a given input datum, output a ranked list of labels along with the unnormalized probability
// /// of that label.
// fn make_prediction_proba<F: Float, L: Label>(
//     model: &ExtraTrees<F, L>,
//     x: &ArrayBase<impl Data<Elem = F>, Ix1>,
// ) -> Vec<(L, f32)> {
//     let prediction_frequencies = get_prediction_freqs(model, x);
//     let mut prediction_vec = prediction_frequencies.into_iter().collect::<Vec<_>>();
//     prediction_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
//     prediction_vec
// }

/// For a given input datum, output a ranked list of labels.
fn make_ranked_prediction<F: Float, L: Label>(
    model: &ExtraTrees<F, L>,
    x: &ArrayBase<impl Data<Elem = F>, Ix1>,
) -> Vec<L> {
    let prediction_frequencies = get_prediction_freqs(model, x);
    let mut keys = prediction_frequencies.keys().cloned().collect::<Vec<_>>();
    keys.sort_by(|a, b| {
        prediction_frequencies[a]
            .partial_cmp(&prediction_frequencies[b])
            .unwrap()
    });
    keys
}
