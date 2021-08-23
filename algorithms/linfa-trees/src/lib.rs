//!
//! # Tree based learning
//! `linfa-trees` aims to provide pure rust implementations
//! of decison tree and forest learning algorithms.
//!
//! # The big picture
//!
//! `linfa-trees` is a crate in the [linfa](https://github.com/rust-ml/linfa) ecosystem,
//! an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's scikit-learn.
//!
//! Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression.
//! The goal is to create a model that predicts the value of a target variable by learning simple decision rules
//! inferred from the data features.
//! 
//! Extremely Randomized Trees (Extra Trees) is an ensemble learning method for classification and regression. It uses
//! many decision trees and randomized splits to achieve improved accuracy and performance.
//!
//! # Current state
//!
//! `linfa-trees` currently provides an [implementation](struct.DecisionTree.html) of single-tree fitting for classification
//! as well as an [implementation](struct.ExtraTrees.html) of Extremely Randomized Trees for classification.
//!

mod decision_trees;
mod extra_trees;

pub use decision_trees::*;
pub use extra_trees::*;
pub use linfa::error::Result;
