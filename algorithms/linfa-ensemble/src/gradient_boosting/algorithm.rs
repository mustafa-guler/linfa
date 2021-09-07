// TODO: Currently clone of bagging. Make this actually gradient boost
use ndarray::{Array2, Array1, ArrayBase, Data, Ix2};

use super::hyperparameters::*;
use linfa::{
    dataset::{AsTargets, Labels},
    error::Error,
    error::Result,
    traits::*,
    DatasetBase, Float, Label,
};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]

#[derive(Debug)]
pub struct GradBoost<F: Float, L: Label, O> 
where
   O: Fit<Array2<F>, Array1<L>, linfa::Error>,
   O::Object: PredictInplace<Array2<F>, Array1<L>>
{
    ensemble: Vec<O::Object>,
}

impl<F: Float, L: Label + std::fmt::Debug, O> GradBoost<F, L, O>
where
   O: Fit<Array2<F>, Array1<L>, linfa::Error>,
   O::Object: PredictInplace<Array2<F>, Array1<L>>,
{
    /// Defaults are provided if the optional parameters are not specified:
    /// * `num_estimators = 1`. 
    /// * `max_n_rows = None`.
    /// The `max_n_rows` default of `None` will be overwritten to the number of rows in the provided dataset. Thus, our bootstrapped data
    /// sets will have the same size as our input data set.
    pub fn params(estimator_params: O) -> GradBoostParams<O> {
        GradBoostParams {
            num_estimators: 1,
            max_n_rows: None,
            estimator_params,
        }
    }
}

impl<'a, F: Float, L: Label + 'a + std::fmt::Debug, O, D, T> Fit<ArrayBase<D, Ix2>, T, Error>
    for GradBoostParams<O>
where
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>,
    O: Fit<Array2<F>, Array1<L>, linfa::Error>,
    O::Object: PredictInplace<Array2<F>, Array1<L>>,
{
    type Object = GradBoost<O>;

    /// Fit bagged ensemble of `O`'s predictors using hyperparameters in `self` on the `dataset`
    /// consisting of a matrix of features and an array of labels.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let num_rows = dataset.records().nrows();
        // self.validate(num_rows)?;   <-- Uncomment if the validation ends up being nontrivial

        // Overrides the `max_n_rows` hyperparameter once the dataset is provided
        // to the actual number of rows if it is `None`
        let true_max_rows = self
            .max_n_rows
            .unwrap_or_else(|| num_rows);

        let mut ensemble = Vec::with_capacity(self.num_estimators);
        let bootstrapper = dataset.bootstrap_samples(true_max_rows, rand::thread_rng());

        // Create all weak learners in the ensemble
        for _ in 0..self.num_estimators {
            // Sample for bootstrapped dataset
            let curr_dataset = bootstrapper.next().unwrap();

            let member = self.estimator_params.fit(curr_dataset)?;
            ensemble.push(member);
        }

        Ok(GradBoost {
            ensemble,
        })
    }
}

impl<F: Float, L: Label + Default, D: Data<Elem = F>, O> PredictInplace<ArrayBase<D, Ix2>, Array1<L>>
    for GradBoost<F, L, O>
where
    O: Fit<Array2<F>, Array1<L>, linfa::Error>,
    O::Object: PredictInplace<Array2<F>, Array1<L>>,
{
    /// Make predictions for each row of a matrix of features `x`.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        let mut all_predictions = vec![];
        for member in self.ensemble {
            let mut target = self.default_target(x);
            member.predict_inplace(x, &target);
            all_predictions.push(target);
        }

        for (ind, target) in y.iter_mut().enumerate() {
            // We now do this super cache-unfriendly operation. Someone let me know if there's a better way to do this
            // that works with the trait bounds.
            let mut prediction_frequencies: HashMap<L, usize> = HashMap::default();
            for member_output in all_predictions {
                let prediction = member_output[ind];
                let value = prediction_frequencies.entry(prediction).or_default();
                *value += 1;
            }


            *target = find_modal_class(&prediction_frequencies);
        }
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}

/// Finds the most frequent class for a hash map of frequencies. If two
/// classes have the same weight then the first class found with that
/// frequency is returned.
fn find_modal_class<L: Label>(class_freq: &HashMap<L, usize>) -> L {
    let val = class_freq
        .iter()
        .fold(None, |acc, (idx, freq)| match acc {
            None => Some((idx, freq)),
            Some((_best_idx, best_freq)) => {
                if best_freq > freq {
                    acc
                } else {
                    Some((idx, freq))
                }
            }
        })
        .unwrap()
        .0;

    (*val).clone()
}
