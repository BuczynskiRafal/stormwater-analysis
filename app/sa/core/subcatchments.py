"""
Feature engineering services for subcatchments.
"""

import logging

import pandas as pd
import swmmio as sw

from .constants import SUBCATCHMENT_CATEGORIES

logger = logging.getLogger(__name__)


class SubcatchmentFeatureEngineeringService:
    def __init__(self, dfs: pd.DataFrame, model: sw.Model):
        """Initialize the service with subcatchment dataframe and SWMM model.

        Args:
            dfs (pd.DataFrame): DataFrame containing subcatchment data.
            model (sw.Model): SWMM model instance.
        """
        self.dfs = dfs
        self.model = model

    def encode_category_column(self, category_column: str = "category") -> pd.DataFrame:
        """One-hot encodes the category column for use in machine learning models.

        Creates columns for all possible categories, not just those present in the data,
        ensuring consistent feature sets across different datasets.

        Args:
            category_column (str, optional): Name of the category column to encode.
                                            Defaults to "category".

        Returns:
            pd.DataFrame: DataFrame with one-hot encoded category columns added
        """
        if self.dfs is None or len(self.dfs) == 0:
            return self.dfs

        if category_column not in self.dfs.columns:
            raise ValueError(f"Column '{category_column}' not found in dataframe")

        encoded_categories = pd.get_dummies(self.dfs[category_column], prefix=None, drop_first=False)

        for category in SUBCATCHMENT_CATEGORIES:
            if category not in encoded_categories.columns:
                encoded_categories[category] = 0
            else:
                encoded_categories[category] = encoded_categories[category].astype(int)

        result_df = pd.concat([self.dfs, encoded_categories], axis=1)
        return result_df

    def subcatchments_classify(self, categories: bool = True) -> None:
        """Assigns subcatchment categories from the TAGS section.

        Gets the category for each subcatchment from the TAGS section of the INP file.
        The category is stored as a tag for each subcatchment.

        Args:
            categories (bool, optional): Not used in this implementation as we always use string categories.
                                        Kept for backward compatibility.
                                        Defaults to True.

        Returns:
            None: Adds 'category' column to the dataframe in-place.
        """
        if self.dfs is None or len(self.dfs) == 0:
            return

        # Alternative: Get tags from the model (for testing purposes)
        # Uncomment this section to use tag-based classification instead of ML classifier
        if hasattr(self.model.inp, "tags") and self.model.inp.tags is not None:
            # Filter tags for subcatchments
            subcatch_tags = self.model.inp.tags[self.model.inp.tags.index.str.startswith("Subcatch")]

            if not subcatch_tags.empty:
                # Create a mapping of subcatchment names to their categories
                category_map = dict(zip(subcatch_tags["Name"], subcatch_tags["Tag"]))

                # Assign categories to subcatchments
                self.dfs["category"] = self.dfs.index.map(category_map)

                # Log the number of categorized subcatchments
                categorized_count = self.dfs["category"].notna().sum()
                logger.info(f"Assigned categories to {categorized_count} subcatchments from TAGS section")
                return  # Exit early if tags were successfully processed
            else:
                logger.warning("No subcatchment tags found in TAGS section")
        else:
            logger.warning("No TAGS section found in the model")

        # Default: Use ML classifier for subcatchment classification

        # required_cols = [
        #     "PercImperv",
        #     "PercSlope",
        #     "N-Imperv",
        #     "N-Perv",
        #     "S-Imperv",
        #     "S-Perv",
        #     "PctZero",
        #     "RunoffCoeff",
        #     "TotalInfil",
        #     "ImpervRunoff",
        #     "TotalRunoffMG",
        #     "PeakRunoff",
        #     "Area",
        # ]

        # df = self.dfs[required_cols].copy()

        # # Convert numeric columns
        # numeric_cols = ["TotalInfil", "ImpervRunoff", "TotalRunoffMG", "PeakRunoff"]
        # for col in numeric_cols:
        #     df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # # Normalize by area (as done in training)
        # df["TotalInfil"] = df["TotalInfil"] / df["Area"]
        # df["ImpervRunoff"] = df["ImpervRunoff"] / df["Area"]
        # df["TotalRunoffMG"] = df["TotalRunoffMG"] / df["Area"]
        # df["PeakRunoff"] = df["PeakRunoff"] / df["Area"]

        # # Select features in the same order as training
        # feature_cols = [
        #     "PercImperv",
        #     "PercSlope",
        #     "N-Imperv",
        #     "N-Perv",
        #     "S-Imperv",
        #     "S-Perv",
        #     "PctZero",
        #     "RunoffCoeff",
        #     "TotalInfil",
        #     "ImpervRunoff",
        #     "TotalRunoffMG",
        #     "PeakRunoff",
        # ]
        # features = df[feature_cols]

        # preds = classifier.predict(features)
        # preds_cls = preds.argmax(axis=-1)

        # if categories:
        #     labels = [
        #         "marshes",
        #         "arable",
        #         "meadows",
        #         "forests",
        #         "rural",
        #         "suburban_weakly_impervious",
        #         "suburban_highly_impervious",
        #         "urban_weakly_impervious",
        #         "urban_moderately_impervious",
        #         "urban_highly_impervious",
        #         "mountains_rocky",
        #         "mountains_vegetated",
        #     ]
        #     self.dfs["category"] = [labels[i] for i in preds_cls]
        # else:
        #     self.dfs["category"] = preds_cls
