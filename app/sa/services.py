"""
Services for data persistence and calculation management.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
from django.db import transaction
from django.utils import timezone

from .models import CalculationSession, ConduitData, NodeData, SubcatchmentData, SWMMModel
from .core.data import DataManager

logger = logging.getLogger(__name__)


class CalculationPersistenceService:
    """Service for saving calculation results to the database."""

    @staticmethod
    def create_session(user, swmm_model: SWMMModel, frost_zone: float) -> CalculationSession:
        """Create a new calculation session."""
        return CalculationSession.objects.create(user=user, swmm_model=swmm_model, frost_zone=frost_zone, status="processing")

    @staticmethod
    def save_calculation_results(session: CalculationSession, data_manager: DataManager) -> bool:
        """
        Save calculation results from DataManager to database.

        Args:
            session: CalculationSession instance
            data_manager: DataManager instance with calculated data

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with transaction.atomic():
                # Save conduit data
                CalculationPersistenceService._save_conduit_data(session, data_manager.dfc)

                # Save node data
                CalculationPersistenceService._save_node_data(session, data_manager.dfn)

                # Save subcatchment data
                CalculationPersistenceService._save_subcatchment_data(session, data_manager.dfs)

                # Update session status
                session.status = "completed"
                session.updated_at = timezone.now()
                session.save()

                logger.info(f"Successfully saved calculation results for session {session.id}")
                return True

        except Exception as e:
            logger.error(f"Error saving calculation results for session {session.id}: {str(e)}")
            session.status = "failed"
            session.error_message = str(e)
            session.save()
            return False

    @staticmethod
    def _save_conduit_data(session: CalculationSession, dfc: pd.DataFrame) -> None:
        """Save conduit data to database."""
        conduit_objects = []

        for idx, row in dfc.iterrows():
            try:
                conduit_obj = ConduitData(
                    session=session,
                    conduit_name=str(idx),
                    geom1=float(row.get("Geom1", 0)),
                    max_v=float(row.get("MaxV", 0)),
                    max_q=float(row.get("MaxQ", 0)),
                    filling=float(row.get("Filling", 0)),
                    slope_per_mile=float(row.get("SlopePerMile", 0)),
                    length=float(row.get("Length", 0)),
                    val_max_fill=int(row.get("ValMaxFill", 0)),
                    val_max_v=int(row.get("ValMaxV", 0)),
                    val_min_v=int(row.get("ValMinV", 0)),
                    val_max_slope=int(row.get("ValMaxSlope", 0)),
                    val_min_slope=int(row.get("ValMinSlope", 0)),
                    val_depth=int(row.get("ValDepth", 0)),
                    val_coverage=int(row.get("ValCoverage", 0)),
                    min_diameter=float(row.get("MinDiameter", 0)),
                    is_min_diameter=int(row.get("isMinDiameter", 0)),
                    increase_dia=int(row.get("IncreaseDia", 0)),
                    reduce_dia=int(row.get("ReduceDia", 0)),
                    increase_slope=int(row.get("IncreaseSlope", 0)),
                    reduce_slope=int(row.get("ReduceSlope", 0)),
                    inlet_node=str(row.get("InletNode", "")),
                    outlet_node=str(row.get("OutletNode", "")),
                    inlet_max_depth=float(row.get("InletMaxDepth", 0)),
                    outlet_max_depth=float(row.get("OutletMaxDepth", 0)),
                    inlet_ground_elevation=float(row.get("InletGroundElevation", 0)),
                    outlet_ground_elevation=float(row.get("OutletGroundElevation", 0)),
                    inlet_ground_cover=float(row.get("InletGroundCover", 0)),
                    outlet_ground_cover=float(row.get("OutletGroundCover", 0)),
                    subcatchment=str(row.get("Subcatchment", "-")),
                    sbc_category=str(row.get("SbcCategory", "-")),
                    recommendation=str(row.get("recommendation", "")),
                )
                conduit_objects.append(conduit_obj)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error processing conduit {idx}: {str(e)}")
                continue

        if conduit_objects:
            ConduitData.objects.bulk_create(conduit_objects, batch_size=1000)
            logger.info(f"Saved {len(conduit_objects)} conduit records")

    @staticmethod
    def _save_node_data(session: CalculationSession, dfn: pd.DataFrame) -> None:
        """Save node data to database."""
        node_objects = []

        for idx, row in dfn.iterrows():
            try:
                node_obj = NodeData(
                    session=session,
                    node_name=str(idx),
                    max_depth=float(row.get("MaxDepth", 0)),
                    invert_elevation=float(row.get("InvertElev", 0)) if "InvertElev" in row else None,
                    subcatchment=str(row.get("Subcatchment", "-")),
                    sbc_category=str(row.get("SbcCategory", "-")),
                )
                node_objects.append(node_obj)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error processing node {idx}: {str(e)}")
                continue

        if node_objects:
            NodeData.objects.bulk_create(node_objects, batch_size=1000)
            logger.info(f"Saved {len(node_objects)} node records")

    @staticmethod
    def _save_subcatchment_data(session: CalculationSession, dfs: pd.DataFrame) -> None:
        """Save subcatchment data to database."""
        subcatchment_objects = []

        for idx, row in dfs.iterrows():
            try:
                subcatchment_obj = SubcatchmentData(
                    session=session,
                    subcatchment_name=str(idx),
                    area=float(row.get("Area", 0)),
                    perc_imperv=float(row.get("PercImperv", 0)),
                    perc_slope=float(row.get("PercSlope", 0)),
                    outlet=str(row.get("Outlet", "")),
                    total_runoff_mg=float(row.get("TotalRunoffMG", 0)) if "TotalRunoffMG" in row else None,
                    peak_runoff=float(row.get("PeakRunoff", 0)) if "PeakRunoff" in row else None,
                    runoff_coeff=float(row.get("RunoffCoeff", 0)) if "RunoffCoeff" in row else None,
                    category=str(row.get("category", "-")),
                )
                subcatchment_objects.append(subcatchment_obj)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error processing subcatchment {idx}: {str(e)}")
                continue

        if subcatchment_objects:
            SubcatchmentData.objects.bulk_create(subcatchment_objects, batch_size=1000)
            logger.info(f"Saved {len(subcatchment_objects)} subcatchment records")


class CalculationRetrievalService:
    """Service for retrieving calculation results from the database."""

    @staticmethod
    def get_session_data(session: CalculationSession) -> Dict[str, Any]:
        """
        Retrieve all data for a calculation session.

        Args:
            session: CalculationSession instance

        Returns:
            Dict containing conduits, nodes, and subcatchments data
        """
        return {
            "session": session,
            "conduits": list(session.conduits.all().values()),
            "nodes": list(session.nodes.all().values()),
            "subcatchments": list(session.subcatchments.all().values()),
        }

    @staticmethod
    def get_user_sessions(user, limit: Optional[int] = None) -> list:
        """
        Get calculation sessions for a user.

        Args:
            user: User instance
            limit: Maximum number of sessions to return

        Returns:
            List of CalculationSession instances
        """
        queryset = CalculationSession.objects.filter(user=user).order_by("-created_at")
        if limit:
            queryset = queryset[:limit]
        return list(queryset)

    @staticmethod
    def format_session_data_for_template(session: CalculationSession) -> Dict[str, list]:
        """
        Format session data for template display.

        Args:
            session: CalculationSession instance

        Returns:
            Dict formatted for template rendering
        """
        conduits_data = []
        for conduit in session.conduits.all():
            conduits_data.append(
                {
                    "Name": conduit.conduit_name,
                    "Diameter": conduit.geom1,
                    "MaxV": conduit.max_v,
                    "MaxQ": conduit.max_q,
                    "Filling": conduit.filling,
                    "Recommendation": conduit.recommendation,
                    "InletNode": conduit.inlet_node,
                    "OutletNode": conduit.outlet_node,
                    "Subcatchment": conduit.subcatchment,
                    "Category": conduit.sbc_category,
                }
            )

        nodes_data = []
        for node in session.nodes.all():
            nodes_data.append(
                {
                    "Name": node.node_name,
                    "MaxDepth": node.max_depth,
                    "InvertElev": node.invert_elevation,
                    "Subcatchment": node.subcatchment,
                    "Category": node.sbc_category,
                }
            )

        subcatchments_data = []
        for subcatchment in session.subcatchments.all():
            subcatchments_data.append(
                {
                    "Name": subcatchment.subcatchment_name,
                    "Area": subcatchment.area,
                    "PercImperv": subcatchment.perc_imperv,
                    "PercSlope": subcatchment.perc_slope,
                    "Outlet": subcatchment.outlet,
                    "Category": subcatchment.category,
                }
            )

        return {
            "Conduits Data": conduits_data,
            "Nodes Data": nodes_data,
            "Subcatchments Data": subcatchments_data,
        }
