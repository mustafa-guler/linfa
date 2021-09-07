//!
//! # Machine Learning Ensemble Methods
//! `linfa-ensemble` aims to provide pure rust implementations
//! of common ensemble methods for machine learning, the most notable being
//! bagging and gradient boosting.
//!
//! # The big picture
//!
//! `linfa-ensemble` is a crate in the [linfa](https://github.com/rust-ml/linfa) ecosystem,
//! an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's scikit-learn.
//!
//! Ensemble methods work by combining assembling a pool (ensemble) of weak learners and then combining their predictions
//! to arrive at a highly predictive model.
//!
//! When training a model via bagging, we sample for selections of our data and train many weak models on the independent subsets.
//! We then predict on a new example by taking the majority vote of our weak learners.
//!
//! For gradient boosting, we iteratively update our model by training a learner on the residual set of our current model. The process
//! is very akin to gradient descent but in a more gneralized function space.
//!
//! # Current state
//!
//! `linfa-ensemble` currently provides an [implementation](struct.Bagging.html) of bagging with a provided classifier,
//! as well as an [implementation](struct.GradBoost.html) of Gradient Boosting.
//!

mod bagging;
//mod gradient_boosting;

pub use bagging::*;
//pub use gradient_boosting::*;
pub use linfa::error::Result;
