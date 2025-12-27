"""
Services for data persistence and calculation management.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
from django.db import transaction
from django.utils import timezone

from accounts.models import User
from .models import CalculationSession, ConduitData, NodeData, SubcatchmentData, SWMMModel
from .core.data import DataManager

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service responsible for running SWMM analysis."""

    @staticmethod
    def analyze_file(
        user: User,
        file_path: str,
        frost_zone: float,
        swmm_model: SWMMModel,
    ) -> tuple[CalculationSession, bool]:
        """
        Run SWMM analysis on a file and save results.

        Args:
            user: User who owns the analysis
            file_path: Path to the SWMM .inp file
            frost_zone: Frost zone value (0.8-1.6)
            swmm_model: SWMMModel instance

        Returns:
            Tuple of (session, success) where success indicates if analysis completed
        """
        session = CalculationPersistenceService.create_session(user=user, swmm_model=swmm_model, frost_zone=frost_zone)

        try:
            with DataManager(file_path, zone=frost_zone) as model:
                success = CalculationPersistenceService.save_calculation_results(session, model)
                return session, success
        except Exception as e:
            logger.error(f"Analysis error for session {session.id}: {e}")
            session.status = "failed"
            session.error_message = str(e)
            session.save()
            return session, False


class CalculationPersistenceService:
    """Service for saving calculation results to the database."""

    @staticmethod
    def _safe_float(value, default=0.0):
        """
        Safely convert value to float, handling None, NaN, and invalid values.
        """
        import math

        if value is None:
            return default
        if isinstance(value, (int, float)):
            if math.isnan(value) if isinstance(value, float) else False:
                return default
            return float(value)
        try:
            result = float(value)
            return result if not math.isnan(result) else default
        except (ValueError, TypeError):
            return default

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
                    min_required_slope=float(row.get("MinRequiredSlope", 0)),
                    increase_slope=int(row.get("IncreaseSlope", 0)),
                    max_allowable_slope=float(row.get("MaxAllowableSlope", 0)),
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
                    # Confidence scores - handle NaN and None values
                    confidence_pump=CalculationPersistenceService._safe_float(row.get("confidence_pump"), 0),
                    confidence_tank=CalculationPersistenceService._safe_float(row.get("confidence_tank"), 0),
                    confidence_seepage_boxes=CalculationPersistenceService._safe_float(row.get("confidence_seepage_boxes"), 0),
                    confidence_diameter_increase=CalculationPersistenceService._safe_float(
                        row.get("confidence_diameter_increase"), 0
                    ),
                    confidence_diameter_reduction=CalculationPersistenceService._safe_float(
                        row.get("confidence_diameter_reduction"), 0
                    ),
                    confidence_slope_increase=CalculationPersistenceService._safe_float(row.get("confidence_slope_increase"), 0),
                    confidence_slope_reduction=CalculationPersistenceService._safe_float(
                        row.get("confidence_slope_reduction"), 0
                    ),
                    confidence_depth_increase=CalculationPersistenceService._safe_float(row.get("confidence_depth_increase"), 0),
                    confidence_valid=CalculationPersistenceService._safe_float(row.get("confidence_valid"), 0),
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
    def get_user_sessions(user, limit: Optional[int] = None, status_filter: Optional[str] = None) -> list:
        """
        Get calculation sessions for a user.

        Args:
            user: User instance
            limit: Maximum number of sessions to return
            status_filter: Filter by session status

        Returns:
            List of CalculationSession instances
        """
        queryset = (
            CalculationSession.objects.filter(user=user)
            .select_related("swmm_model")
            .prefetch_related("conduits", "nodes", "subcatchments")
            .order_by("-created_at")
        )

        if status_filter:
            queryset = queryset.filter(status=status_filter)

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


class NavigationService:
    """Service for handling navigation between elements."""

    @staticmethod
    def get_navigation_context(session: CalculationSession, element_type: str, element_name: str) -> Dict[str, Any]:
        """
        Get navigation context for an element (previous/next URLs).

        Args:
            session: CalculationSession instance
            element_type: Type of element ('conduit', 'node', 'subcatchment')
            element_name: Name of the current element

        Returns:
            Dict containing previous and next element names and URLs
        """
        if element_type == "conduit":
            return NavigationService._get_conduit_navigation(session, element_name)
        elif element_type == "node":
            return NavigationService._get_node_navigation(session, element_name)
        elif element_type == "subcatchment":
            return NavigationService._get_subcatchment_navigation(session, element_name)
        else:
            return {"previous": None, "next": None}

    @staticmethod
    def _get_conduit_navigation(session: CalculationSession, current_name: str) -> Dict[str, Any]:
        """Get navigation context for conduits."""
        conduits = list(session.conduits.all().order_by("id"))
        conduit_names = [c.conduit_name for c in conduits]

        try:
            current_index = conduit_names.index(current_name)
            previous_name = conduit_names[current_index - 1] if current_index > 0 else None
            next_name = conduit_names[current_index + 1] if current_index < len(conduit_names) - 1 else None

            return {
                "previous": {
                    "name": previous_name,
                    "url": f"/analysis/results/{session.id}/conduit/{previous_name}/" if previous_name else None,
                },
                "next": {"name": next_name, "url": f"/analysis/results/{session.id}/conduit/{next_name}/" if next_name else None},
            }
        except ValueError:
            return {"previous": None, "next": None}

    @staticmethod
    def _get_node_navigation(session: CalculationSession, current_name: str) -> Dict[str, Any]:
        """Get navigation context for nodes."""
        nodes = list(session.nodes.all().order_by("id"))
        node_names = [n.node_name for n in nodes]

        try:
            current_index = node_names.index(current_name)
            previous_name = node_names[current_index - 1] if current_index > 0 else None
            next_name = node_names[current_index + 1] if current_index < len(node_names) - 1 else None

            return {
                "previous": {
                    "name": previous_name,
                    "url": f"/analysis/results/{session.id}/node/{previous_name}/" if previous_name else None,
                },
                "next": {"name": next_name, "url": f"/analysis/results/{session.id}/node/{next_name}/" if next_name else None},
            }
        except ValueError:
            return {"previous": None, "next": None}

    @staticmethod
    def _get_subcatchment_navigation(session: CalculationSession, current_name: str) -> Dict[str, Any]:
        """Get navigation context for subcatchments."""
        subcatchments = list(session.subcatchments.all().order_by("id"))
        subcatchment_names = [s.subcatchment_name for s in subcatchments]

        try:
            current_index = subcatchment_names.index(current_name)
            previous_name = subcatchment_names[current_index - 1] if current_index > 0 else None
            next_name = subcatchment_names[current_index + 1] if current_index < len(subcatchment_names) - 1 else None

            return {
                "previous": {
                    "name": previous_name,
                    "url": f"/analysis/results/{session.id}/subcatchment/{previous_name}/" if previous_name else None,
                },
                "next": {
                    "name": next_name,
                    "url": f"/analysis/results/{session.id}/subcatchment/{next_name}/" if next_name else None,
                },
            }
        except ValueError:
            return {"previous": None, "next": None}
