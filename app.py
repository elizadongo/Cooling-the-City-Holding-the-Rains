
from pathlib import Path
import os
import warnings
import importlib.util
import math

# ============================================================
# 0. PROJ CONFIGURATION
# ============================================================

def configure_proj():
    candidates = []
    gdal_candidates = []

    for package_name in ("pyproj", "rasterio"):
        spec = importlib.util.find_spec(package_name)

        if spec is None or spec.origin is None:
            continue

        package_dir = Path(spec.origin).resolve().parent

        for database in package_dir.rglob("proj.db"):
            candidates.append(database.parent)

        if package_name == "rasterio":
            for csv_file in package_dir.rglob("gcs.csv"):
                gdal_candidates.append(csv_file.parent)

    proj_directory = next(
        (
            path for path in candidates
            if (path / "proj.db").is_file()
        ),
        None,
    )

    if proj_directory is None:
        raise RuntimeError(
            "A compatible PROJ database could not be located "
            "inside the installed PyProj or Rasterio packages."
        )

    os.environ["PROJ_DATA"] = str(proj_directory)
    os.environ["PROJ_LIB"] = str(proj_directory)
    os.environ["PROJ_NETWORK"] = "OFF"

    if gdal_candidates:
        os.environ["GDAL_DATA"] = str(gdal_candidates[0])


configure_proj()

# ============================================================
# 1. IMPORTS AND PAGE CONFIGURATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import folium

from shapely.geometry import Point
from folium import LayerControl
from folium.plugins import Fullscreen, MousePosition
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium


st.set_page_config(
    page_title="Cooling the City, Holding the Rains",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "aoi": "Study_AOI_UTM36N.shp",
    "waterways": "Waterways_AOI_UTM36N.shp",
    "wetlands": "Wetlands_AOI_UTM36N.shp",
    "interventions": "Blue_Green_Intervention_Types.tif",
    "intervention_polygons": "Blue_Green_Intervention_Types_Polygons.gpkg",
    "integrated": "Integrated_Blue_Green_Priority_Verified.tif",
    "heat": "Heat_Reduction_Priority_Classes_Proxy.tif",
    "stormwater": "Stormwater_Priority_Classes.tif",
    "waterway_suitability": "Waterway_Proximity_Suitability.tif",
}

# ============================================================
# 2. COLOURS AND SYMBOLOGY
# ============================================================

PRIORITY_COLOURS = {
    1: "#2166AC",
    2: "#67A9CF",
    3: "#FFFF33",
    4: "#F28E2B",
    5: "#D7191C",
}

PRIORITY_LABELS = {
    1: "Very low priority",
    2: "Low priority",
    3: "Moderate priority",
    4: "High priority",
    5: "Very high priority",
}

INTERVENTION_COLOURS = {
    0: "#D9DEE5",
    1: "#238443",
    2: "#2878B5",
    3: "#8C4A9E",
}

INTERVENTION_LABELS = {
    0: "No candidate criterion met",
    1: "Cooling / tree intervention",
    2: "Riparian restoration / protection",
    3: "Integrated blue-green intervention",
}

COLOURS = {
    "blue": "#2878B5",
    "green": "#238443",
    "orange": "#E69F00",
    "red": "#C83E3E",
    "dark": "#263238",
    "light": "#F3F6F4",
    "grey": "#6B7280",
    "purple": "#8C4A9E",
}

# ============================================================
# 3. INTERVENTION SVG SYMBOLS
# ============================================================

# These SVG symbols reproduce the visual meaning of the
# original QGIS intervention categories in the web map.

TREE_SYMBOL = """
<svg xmlns="http://www.w3.org/2000/svg"
     width="28" height="28" viewBox="0 0 28 28">
    <path d="M14 17 L14 26" stroke="#654321"
          stroke-width="2.4" stroke-linecap="round"/>
    <path d="M14 2 L5 15 L10 15 L4 21 L24 21
             L18 15 L23 15 Z"
          fill="#238443" stroke="#145A32"
          stroke-width="1.1" stroke-linejoin="round"/>
    <path d="M14 5 L14 18" stroke="#D7F0D8"
          stroke-width="0.8" opacity="0.6"/>
</svg>
"""

RIPARIAN_SYMBOL = """
<svg xmlns="http://www.w3.org/2000/svg"
     width="32" height="28" viewBox="0 0 32 28">
    <path d="M2 21 Q8 16 14 21 T26 21"
          fill="none" stroke="#2878B5"
          stroke-width="2.5" stroke-linecap="round"/>
    <path d="M3 25 Q9 20 15 25 T29 25"
          fill="none" stroke="#2878B5"
          stroke-width="1.5" stroke-linecap="round"/>
    <path d="M8 19 L8 7 M8 12 L4 9 M8 14 L12 10"
          stroke="#238443" stroke-width="1.8"
          fill="none" stroke-linecap="round"/>
    <path d="M22 19 L22 5 M22 11 L18 8 M22 14 L26 9"
          stroke="#238443" stroke-width="1.8"
          fill="none" stroke-linecap="round"/>
</svg>
"""

INTEGRATED_SYMBOL = """
<svg xmlns="http://www.w3.org/2000/svg"
     width="38" height="30" viewBox="0 0 38 30">
    <path d="M2 25 Q9 20 16 25 T34 25"
          fill="none" stroke="#2878B5"
          stroke-width="2.2" stroke-linecap="round"/>
    <path d="M9 17 L9 25" stroke="#654321"
          stroke-width="2" stroke-linecap="round"/>
    <path d="M9 2 L3 13 L7 13 L3 18 L15 18
             L11 13 L15 13 Z"
          fill="#238443" stroke="#145A32"
          stroke-width="0.8"/>
    <path d="M23 20 L23 12 M23 15 L19 12 M23 16 L27 12"
          stroke="#238443" stroke-width="1.7"
          fill="none" stroke-linecap="round"/>
    <path d="M29 20 L29 13 M29 16 L26 14 M29 17 L32 14"
          stroke="#238443" stroke-width="1.5"
          fill="none" stroke-linecap="round"/>
    <path d="M18 27 Q24 23 30 27"
          fill="none" stroke="#2878B5"
          stroke-width="1.5"/>
</svg>
"""

INTERVENTION_SYMBOLS = {
    1: TREE_SYMBOL,
    2: RIPARIAN_SYMBOL,
    3: INTEGRATED_SYMBOL,
}

# ============================================================
# 4. FILE AND DATA FUNCTIONS
# ============================================================

def file_path(key):
    return DATA_DIR / FILES[key]


def file_exists(key):
    path = file_path(key)

    if path.suffix.lower() == ".shp":
        required = [".shp", ".shx", ".dbf", ".prj"]
        return all(path.with_suffix(ext).exists() for ext in required)

    return path.exists()


def show_missing_files(missing):
    st.error("Some project data files could not be found.")

    st.markdown(
        "Check that the following files are inside the dashboard's "
        "`data` folder:"
    )

    for name in missing:
        st.code(str(DATA_DIR / name), language=None)

    st.info(
        "Keep all shapefile components together. Check that the "
        "GeoPackage and raster filenames match the project filenames."
    )


@st.cache_data(show_spinner=False)
def load_vector(path_string):
    gdf = gpd.read_file(path_string)

    if gdf.crs is None:
        raise ValueError(
            f"Coordinate reference system missing: {path_string}"
        )

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    return gdf.to_crs(epsg=4326)


@st.cache_data(show_spinner=False)
def read_raster_info(path_string):
    with rasterio.open(path_string) as src:
        band = src.read(1, masked=True)
        values = band.compressed()
        values = values[np.isfinite(values)]

        if values.size == 0:
            stats = {
                "minimum": None,
                "maximum": None,
                "mean": None,
                "valid_pixels": 0,
            }
        else:
            stats = {
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
                "valid_pixels": int(values.size),
            }

        return {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "resolution": src.res,
            "nodata": src.nodata,
            "bounds": tuple(src.bounds),
            "dtype": str(src.dtypes[0]),
            "stats": stats,
        }


@st.cache_data(show_spinner=False)
def read_raster_array(path_string):
    with rasterio.open(path_string) as src:
        array = src.read(1, masked=True).astype("float32")

        return (
            np.ma.filled(array, np.nan),
            tuple(src.bounds),
            src.crs.to_string() if src.crs else None,
            src.nodata,
            src.transform,
            src.width,
            src.height,
        )


def raster_bounds_wgs84(bounds, crs):
    if not crs:
        raise ValueError("Raster coordinate reference system is missing.")

    left, bottom, right, top = bounds

    west, south, east, north = transform_bounds(
        crs,
        "EPSG:4326",
        left,
        bottom,
        right,
        top,
        densify_pts=21,
    )

    return [[south, west], [north, east]]


def hex_to_rgb(colour):
    colour = colour.lstrip("#")

    return tuple(
        int(colour[i:i + 2], 16)
        for i in (0, 2, 4)
    )


def make_rgba(
    array,
    palette,
    opacity=165,
    categorical=True,
    continuous_min=1,
    continuous_max=5,
):
    data = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(data)

    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)

    if not valid.any():
        return rgba

    if categorical:
        rounded = np.rint(data)

        for value, colour in palette.items():
            rgb = hex_to_rgb(colour)
            mask = valid & (rounded == value)

            rgba[mask, 0] = rgb[0]
            rgba[mask, 1] = rgb[1]
            rgba[mask, 2] = rgb[2]
            rgba[mask, 3] = opacity

    else:
        clipped = np.clip(
            data,
            continuous_min,
            continuous_max,
        )

        denominator = continuous_max - continuous_min

        if denominator <= 0:
            normalised = np.zeros_like(clipped)
        else:
            normalised = (
                (clipped - continuous_min) / denominator
            )

        stops = [
            (0.00, hex_to_rgb("#2166AC")),
            (0.25, hex_to_rgb("#67A9CF")),
            (0.50, hex_to_rgb("#FFFF33")),
            (0.75, hex_to_rgb("#F28E2B")),
            (1.00, hex_to_rgb("#D7191C")),
        ]

        for index in range(len(stops) - 1):
            p0, c0 = stops[index]
            p1, c1 = stops[index + 1]

            if index == len(stops) - 2:
                mask = (
                    valid
                    & (normalised >= p0)
                    & (normalised <= p1)
                )
            else:
                mask = (
                    valid
                    & (normalised >= p0)
                    & (normalised < p1)
                )

            if not mask.any():
                continue

            fraction = (
                (normalised[mask] - p0) / (p1 - p0)
            )

            for channel in range(3):
                rgba[mask, channel] = np.clip(
                    c0[channel]
                    + fraction * (c1[channel] - c0[channel]),
                    0,
                    255,
                ).astype(np.uint8)

        rgba[valid, 3] = opacity

    return rgba


def add_raster_overlay(
    fmap,
    path_string,
    layer_name,
    palette,
    opacity=0.65,
    categorical=True,
    continuous_min=1,
    continuous_max=5,
    show=False,
):
    array, bounds, crs, _, _, _, _ = read_raster_array(path_string)

    rgba = make_rgba(
        array,
        palette,
        opacity=int(opacity * 255),
        categorical=categorical,
        continuous_min=continuous_min,
        continuous_max=continuous_max,
    )

    image_bounds = raster_bounds_wgs84(bounds, crs)

    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=image_bounds,
        name=layer_name,
        opacity=1.0,
        interactive=True,
        cross_origin=False,
        zindex=2,
        show=show,
    ).add_to(fmap)


def add_vector_layer(
    fmap,
    gdf,
    name,
    colour,
    weight=2,
    fill_opacity=0.12,
    show=True,
):
    if gdf is None or gdf.empty:
        return

    fields = [
        column for column in gdf.columns
        if column != "geometry"
    ][:3]

    tooltip = None

    if fields:
        tooltip = folium.GeoJsonTooltip(
            fields=fields,
            aliases=[
                field.replace("_", " ").title()
                for field in fields
            ],
            sticky=True,
        )

    folium.GeoJson(
        gdf.__geo_interface__,
        name=name,
        style_function=lambda feature: {
            "color": colour,
            "weight": weight,
            "fillColor": colour,
            "fillOpacity": fill_opacity,
        },
        highlight_function=lambda feature: {
            "weight": weight + 0.5,
            "color": "#111827",
            "fillOpacity": min(fill_opacity + 0.08, 0.35),
        },
        tooltip=tooltip,
        show=show,
    ).add_to(fmap)


# ============================================================
# 5. INTERVENTION POLYGON AND PATTERN SYMBOL FUNCTIONS
# ============================================================

def intervention_category(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def add_intervention_polygon_layer(fmap, gdf, show=True):
    if gdf is None or gdf.empty:
        return

    if "DN" not in gdf.columns:
        raise ValueError(
            "The intervention polygon layer does not contain DN."
        )

    display_gdf = gdf.copy()

    display_gdf["Intervention_Type"] = (
        pd.to_numeric(display_gdf["DN"], errors="coerce")
        .map(INTERVENTION_LABELS)
        .fillna("Unclassified intervention")
    )

    def category(feature):
        value = feature.get("properties", {}).get("DN")
        return intervention_category(value)

    def style_function(feature):
        value = category(feature)

        colour = INTERVENTION_COLOURS.get(
            value,
            "#D9DEE5",
        )

        return {
            "color": colour,
            "weight": 0.5,
            "opacity": 0.85,
            "fillColor": colour,
            "fillOpacity": 0.35,
        }

    def highlight_function(feature):
        value = category(feature)

        colour = INTERVENTION_COLOURS.get(
            value,
            "#D9DEE5",
        )

        return {
            "color": "#17202A",
            "weight": 1.5,
            "opacity": 1,
            "fillColor": colour,
            "fillOpacity": 0.55,
        }

    folium.GeoJson(
        display_gdf.__geo_interface__,
        name="Proposed blue-green intervention types",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["DN", "Intervention_Type"],
            aliases=["Category:", "Intervention type:"],
            sticky=True,
            labels=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=["DN", "Intervention_Type"],
            aliases=["Category:", "Intervention type:"],
            labels=True,
        ),
        show=show,
    ).add_to(fmap)


def add_intervention_pattern_symbols(
    fmap,
    gdf,
    spacing_m=110,
):
    """
    Place repeated intervention symbols within the intervention
    polygons using a projected metre-based grid.

    The original QGIS point-pattern appearance is represented
    visually, but the spacing is geographic rather than a fixed
    4 mm screen/map-layout spacing.
    """

    if gdf is None or gdf.empty:
        return

    if "DN" not in gdf.columns:
        return

    projected = gdf[["DN", "geometry"]].copy()
    projected = projected.to_crs(epsg=32636)

    projected["DN"] = pd.to_numeric(
        projected["DN"],
        errors="coerce",
    )

    projected = projected[
        projected["DN"].isin([1, 2, 3])
    ].copy()

    projected = projected[
        projected.geometry.notna()
        & ~projected.geometry.is_empty
    ]

    if projected.empty:
        return

    minx, miny, maxx, maxy = projected.total_bounds

    # Align the pattern to a consistent projected grid.
    start_x = math.floor(minx / spacing_m) * spacing_m
    start_y = math.floor(miny / spacing_m) * spacing_m

    grid_points = []

    x_values = np.arange(
        start_x,
        maxx + spacing_m,
        spacing_m,
    )

    y_values = np.arange(
        start_y,
        maxy + spacing_m,
        spacing_m,
    )

    for x in x_values:
        for y in y_values:
            grid_points.append(Point(float(x), float(y)))

    if not grid_points:
        return

    points = gpd.GeoDataFrame(
        {"geometry": grid_points},
        geometry="geometry",
        crs="EPSG:32636",
    )

    # Assign each grid point to the intervention polygon
    # containing it.
    try:
        joined = gpd.sjoin(
            points,
            projected[["DN", "geometry"]],
            how="inner",
            predicate="within",
        )

    except Exception as exc:
        raise RuntimeError(
            "Could not assign pattern symbols to intervention polygons. "
            "Check the GeoPandas spatial-index installation."
        ) from exc

    if joined.empty:
        return

    # Avoid duplicate symbols where polygons overlap.
    joined = joined[
        ~joined.index.duplicated(keep="first")
    ].copy()

    joined = joined.to_crs(epsg=4326)

    # Draw the category-specific symbol at each grid location.
    for _, row in joined.iterrows():

        category = intervention_category(row["DN"])

        symbol = INTERVENTION_SYMBOLS.get(category)

        if symbol is None:
            continue

        marker = folium.Marker(
            location=[
                row.geometry.y,
                row.geometry.x,
            ],
            icon=folium.DivIcon(
                html=(
                    '<div style="'
                    'background:transparent;'
                    'border:none;'
                    'width:40px;'
                    'height:32px;'
                    'display:flex;'
                    'align-items:center;'
                    'justify-content:center;'
                    '">'
                    f'{symbol}'
                    '</div>'
                ),
                icon_size=(40, 32),
                icon_anchor=(20, 16),
                class_name="intervention-pattern-marker",
            ),
            tooltip=INTERVENTION_LABELS[category],
        )

        marker.add_to(fmap)


# ============================================================
# 6. OTHER MAP AND SUMMARY FUNCTIONS
# ============================================================

def get_map_centre(aoi):
    if aoi is not None and not aoi.empty:
        centroid = aoi.geometry.union_all().centroid
        return [centroid.y, centroid.x]

    return [0.3476, 32.5825]


def calculate_vector_summary(gdf, label):
    if gdf is None:
        return {
            "Layer": label,
            "Features": 0,
            "Geometry types": "Not loaded",
        }

    geometry_types = ", ".join(
        sorted(gdf.geometry.geom_type.dropna().unique())
    )

    return {
        "Layer": label,
        "Features": len(gdf),
        "Geometry types": geometry_types or "Unknown",
    }


def format_number(value, decimals=2):
    if value is None or not np.isfinite(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


# ============================================================
# 7. LEGEND FUNCTIONS
# ============================================================

def render_priority_legend(title):
    st.markdown(f"**{title}**")

    for value, label in PRIORITY_LABELS.items():

        colour = PRIORITY_COLOURS[value]

        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:9px;
                margin-bottom:6px;
            ">
                <div style="
                    background:{colour};
                    width:22px;
                    height:17px;
                    border:1px solid #777;
                    border-radius:2px;
                    flex-shrink:0;
                "></div>
                <span>{value} — {label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_continuous_priority_legend():
    st.markdown("**Integrated blue-green priority**")

    st.markdown(
        """
        <div style="
            height:20px;
            border:1px solid #777;
            border-radius:3px;
            background:linear-gradient(
                to right,
                #2166AC,
                #67A9CF,
                #FFFF33,
                #F28E2B,
                #D7191C
            );
        "></div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    left.caption("Lower score")
    right.caption("Higher score")


def render_intervention_legend():
    st.markdown("**Proposed blue-green intervention types**")

    for value, label in INTERVENTION_LABELS.items():

        colour = INTERVENTION_COLOURS[value]

        symbol = INTERVENTION_SYMBOLS.get(value, "")

        if value == 0:
            symbol_display = (
                '<span style="font-size:12px;color:#555;">'
                'No pattern symbol'
                '</span>'
            )
        else:
            symbol_display = (
                f'<div style="width:42px;height:30px;">'
                f'{symbol}</div>'
            )

        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:10px;
                margin-bottom:10px;
                padding:5px 7px;
                border-radius:4px;
                background:rgba(128,128,128,0.06);
            ">
                <div style="
                    width:23px;
                    height:19px;
                    flex-shrink:0;
                    background:{colour};
                    border:1px solid #777;
                    border-radius:2px;
                "></div>
                {symbol_display}
                <span>{value} — {label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Pattern symbols are a web-map representation of the "
        "intervention categories. Their spacing changes with map scale."
    )


def render_selected_map_legend(
    show_integrated,
    show_heat,
    show_stormwater,
    show_waterway_suitability,
    show_interventions,
):
    """
    Display only the thematic legends corresponding to
    the currently selected thematic map layers.
    """

    selected = any(
        [
            show_integrated,
            show_heat,
            show_stormwater,
            show_waterway_suitability,
            show_interventions,
        ]
    )

    st.subheader("Selected layer symbology")

    if not selected:
        st.info(
            "No thematic layers are selected. Select a priority "
            "raster or intervention layer above the map to display "
            "its legend."
        )
        return

    legend_columns = st.columns(2)
    column_index = 0

    if show_integrated:
        with legend_columns[column_index % 2]:
            render_continuous_priority_legend()

        column_index += 1

    if show_heat:
        with legend_columns[column_index % 2]:
            render_priority_legend("Heat-reduction priority")

        column_index += 1

    if show_stormwater:
        with legend_columns[column_index % 2]:
            render_priority_legend("Stormwater priority")

        column_index += 1

    if show_waterway_suitability:
        with legend_columns[column_index % 2]:
            render_priority_legend(
                "Waterway proximity suitability"
            )

        column_index += 1

    if show_interventions:
        with legend_columns[column_index % 2]:
            render_intervention_legend()

        column_index += 1


# ============================================================
# 8. HEADING AND INTRODUCTION
# ============================================================

st.markdown(
    """
    <div style="
        background-color:#183F35;
        color:white;
        padding:24px 28px;
        border-radius:10px;
        margin-bottom:12px;
        border-left:7px solid #79B96B;
    ">
        <h1 style="color:white;margin:0;font-size:2.1rem;">
            🌿 Cooling the City, Holding the Rains
        </h1>
        <p style="color:#E2F0E5;margin:8px 0 0 0;font-size:1rem;">
            GIS-based planning for urban cooling, stormwater management
            and ecological conservation
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    **GIS-based planning of an integrated blue-green infrastructure
    network for heat reduction, stormwater management and ecological
    conservation.**
    """
)

st.caption(
    "Spatial decision-support dashboard | Study area: Study_AOI_UTM36N"
)

# ============================================================
# 9. LOAD PROJECT DATA
# ============================================================

missing_files = [
    filename
    for key, filename in FILES.items()
    if not file_exists(key)
]

if missing_files:
    show_missing_files(missing_files)
    st.stop()


@st.cache_data(show_spinner="Loading study area and spatial layers...")
def load_project_vectors():
    aoi = load_vector(str(file_path("aoi")))
    waterways = load_vector(str(file_path("waterways")))
    wetlands = load_vector(str(file_path("wetlands")))
    intervention_polygons = load_vector(
        str(file_path("intervention_polygons"))
    )

    return aoi, waterways, wetlands, intervention_polygons


try:
    (
        aoi,
        waterways,
        wetlands,
        intervention_polygons,
    ) = load_project_vectors()

    integrated_info = read_raster_info(str(file_path("integrated")))
    heat_info = read_raster_info(str(file_path("heat")))
    storm_info = read_raster_info(str(file_path("stormwater")))
    intervention_info = read_raster_info(str(file_path("interventions")))
    waterway_info = read_raster_info(
        str(file_path("waterway_suitability"))
    )

except Exception as exc:
    st.error("The project data could not be loaded.")
    st.exception(exc)
    st.stop()


# ============================================================
# 10. SIDEBAR: DASHBOARD CONTROLS
# ============================================================

st.sidebar.title("Dashboard controls")

st.sidebar.markdown(
    "Use the dropdown sections below to adjust the map display."
)

with st.sidebar.expander("Map display settings", expanded=True):

    basemap_choice = st.selectbox(
        "Basemap",
        ["OpenStreetMap", "CartoDB positron"],
        index=0,
    )

    overlay_opacity = st.slider(
        "Raster overlay opacity",
        min_value=0.10,
        max_value=1.00,
        value=0.60,
        step=0.05,
    )

    map_height = st.selectbox(
        "Map display height",
        [500, 560, 620, 680],
        index=1,
        format_func=lambda x: f"{x} pixels",
    )

with st.sidebar.expander("Study area information", expanded=False):

    st.write("**Study boundary:** Study_AOI_UTM36N")
    st.write("**Projected CRS:** EPSG:32636")
    st.write(f"**Waterway features:** {len(waterways):,}")
    st.write(f"**Wetland features:** {len(wetlands):,}")

    st.write(
        "**Integrated priority valid pixels:** "
        f"{integrated_info['stats']['valid_pixels']:,}"
    )

with st.sidebar.expander("How to use the dashboard", expanded=False):

    st.markdown(
        """
        1. Open **Map Explorer**.
        2. Use the map layer selection dropdown to choose layers.
        3. Adjust opacity and basemap in the sidebar.
        4. Inspect the statistics and scenario comparisons.
        5. Review the methods and limitations before interpreting results.
        """
    )


# ============================================================
# 11. SUMMARY METRICS
# ============================================================

valid_integrated = integrated_info["stats"]["valid_pixels"]
valid_heat = heat_info["stats"]["valid_pixels"]
valid_storm = storm_info["stats"]["valid_pixels"]

aoi_area_km2 = (
    float(aoi.to_crs(epsg=32636).geometry.area.sum())
    / 1_000_000
)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Study area", f"{aoi_area_km2:.3f} km²")
metric2.metric("Waterway features", f"{len(waterways):,}")
metric3.metric("Wetland features", f"{len(wetlands):,}")
metric4.metric("Integrated valid pixels", f"{valid_integrated:,}")

st.divider()


# ============================================================
# 12. FIVE ORIGINAL DASHBOARD TABS
# ============================================================

(
    tab_overview,
    tab_map,
    tab_stats,
    tab_scenario,
    tab_methods,
) = st.tabs(
    [
        "Study Area Overview",
        "Map Explorer",
        "Statistics",
        "Scenario Lab",
        "Methods & Limitations",
    ]
)


# ============================================================
# TAB 1: STUDY AREA OVERVIEW
# ============================================================

with tab_overview:

    st.header("Study Area Overview")

    st.markdown(
        """
        This dashboard supports the identification of locations where
        blue-green infrastructure may help address urban heat,
        stormwater pressure and ecological conservation.

        The integrated priority surface combines mapped planning
        indicators into a spatial screening layer. It supports further
        investigation and does not replace field verification or
        engineering design.
        """
    )

    col1, col2 = st.columns([1.15, 1])

    with col1:

        st.subheader("Project objectives")

        st.markdown(
            """
            - Identify locations with high integrated blue-green
              infrastructure priority.
            - Explore areas highlighted by heat-reduction and
              stormwater indicators.
            - Examine spatial relationships between proposed
              interventions, waterways and wetlands.
            - Support planning of green corridors, infiltration areas,
              retention spaces and riparian buffers.
            - Provide a starting point for site assessment and
              further spatial analysis.
            """
        )

    with col2:

        st.subheader("Available spatial indicators")

        indicator_data = pd.DataFrame(
            [
                {
                    "Indicator": "Integrated priority",
                    "Valid pixels": valid_integrated,
                    "Mean": integrated_info["stats"]["mean"],
                },
                {
                    "Indicator": "Heat priority",
                    "Valid pixels": valid_heat,
                    "Mean": heat_info["stats"]["mean"],
                },
                {
                    "Indicator": "Stormwater priority",
                    "Valid pixels": valid_storm,
                    "Mean": storm_info["stats"]["mean"],
                },
            ]
        )

        indicator_display = indicator_data.copy()

        indicator_display["Mean"] = indicator_display["Mean"].apply(
            lambda x: format_number(x, 3)
        )

        st.dataframe(
            indicator_display,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Interpretation of priority classes")
    render_priority_legend("Priority classification")

    st.info(
        "Priority classes describe relative priority within the relevant "
        "raster classification. They should not be interpreted as "
        "measured temperature reductions, flood probabilities or "
        "confirmed intervention performance."
    )

    st.subheader("Planning logic")

    st.markdown(
        """
        **Urban pressure → spatial indicators → integrated priority →
        candidate interventions → field validation**

        Priority maps can direct attention to candidate sites. Final
        selection requires consideration of land ownership, drainage,
        soil infiltration, infrastructure and local constraints.
        """
    )


# ============================================================
# TAB 2: MAP EXPLORER
# ============================================================

with tab_map:

    st.header("Map Explorer")

    st.markdown(
        """
        Select the layers you want to view. The map occupies the full
        available width, and the corresponding thematic legend appears
        directly below it.
        """
    )

    # --------------------------------------------------------
    # LAYER SELECTION: ABOVE THE FULL-WIDTH MAP
    # --------------------------------------------------------

    with st.expander("Map layer selection", expanded=True):

        selection_col1, selection_col2, selection_col3 = st.columns(3)

        with selection_col1:

            st.markdown("**Priority raster layers**")

            show_integrated = st.checkbox(
                "Integrated blue-green priority",
                value=True,
                key="legend_integrated",
            )

            show_heat = st.checkbox(
                "Heat-reduction priority",
                value=False,
                key="legend_heat",
            )

            show_stormwater = st.checkbox(
                "Stormwater priority",
                value=False,
                key="legend_stormwater",
            )

            show_waterway_suitability = st.checkbox(
                "Waterway proximity suitability",
                value=False,
                key="legend_waterway_suitability",
            )

        with selection_col2:

            st.markdown("**Intervention layer**")

            show_interventions = st.checkbox(
                "Proposed intervention types",
                value=False,
                key="legend_interventions",
            )

            st.caption(
                "Displays the tree, riparian and integrated "
                "blue-green intervention categories."
            )

        with selection_col3:

            st.markdown("**Reference layers**")

            show_wetlands = st.checkbox(
                "Wetlands",
                value=True,
                key="legend_wetlands",
            )

            show_waterways = st.checkbox(
                "Waterways",
                value=True,
                key="legend_waterways",
            )

            show_aoi = st.checkbox(
                "Study area boundary",
                value=True,
                key="legend_aoi",
            )

    # --------------------------------------------------------
    # MAP CONSTRUCTION: FULL WIDTH
    # --------------------------------------------------------

    centre = get_map_centre(aoi)

    fmap = folium.Map(
        location=centre,
        zoom_start=14,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
        max_zoom=19,
    )

    if basemap_choice == "OpenStreetMap":

        folium.TileLayer(
            tiles="OpenStreetMap",
            name="OpenStreetMap",
            show=True,
        ).add_to(fmap)

    else:

        folium.TileLayer(
            tiles="CartoDB positron",
            name="CartoDB positron",
            show=True,
        ).add_to(fmap)

    # --------------------------------------------------------
    # RASTER OVERLAYS
    # --------------------------------------------------------

    raster_layers = [
        (
            "integrated",
            "Integrated blue-green priority",
            PRIORITY_COLOURS,
            False,
            show_integrated,
        ),
        (
            "heat",
            "Heat-reduction priority",
            PRIORITY_COLOURS,
            True,
            show_heat,
        ),
        (
            "stormwater",
            "Stormwater priority",
            PRIORITY_COLOURS,
            True,
            show_stormwater,
        ),
        (
            "waterway_suitability",
            "Waterway proximity suitability",
            PRIORITY_COLOURS,
            True,
            show_waterway_suitability,
        ),
    ]

    for (
        key,
        layer_name,
        palette,
        categorical,
        layer_visible,
    ) in raster_layers:

        if layer_visible:

            try:

                add_raster_overlay(
                    fmap=fmap,
                    path_string=str(file_path(key)),
                    layer_name=layer_name,
                    palette=palette,
                    opacity=overlay_opacity,
                    categorical=categorical,
                    continuous_min=1,
                    continuous_max=5,
                    show=True,
                )

            except Exception as exc:

                st.error(
                    f"Could not display {layer_name}."
                )
                st.exception(exc)

    # --------------------------------------------------------
    # INTERVENTION POLYGONS AND THEIR PATTERN SYMBOLS
    # --------------------------------------------------------

    if show_interventions:

        try:

            add_intervention_polygon_layer(
                fmap,
                intervention_polygons,
                show=True,
            )

            add_intervention_pattern_symbols(
                fmap,
                intervention_polygons,
                spacing_m=110,
            )

        except Exception as exc:

            st.error(
                "The intervention layer or its pattern symbols "
                "could not be displayed."
            )
            st.exception(exc)

    # --------------------------------------------------------
    # VECTOR REFERENCE LAYERS
    # --------------------------------------------------------

    if show_wetlands:

        add_vector_layer(
            fmap,
            wetlands,
            "Wetlands",
            COLOURS["blue"],
            weight=1.5,
            fill_opacity=0.10,
            show=True,
        )

    if show_waterways:

        add_vector_layer(
            fmap,
            waterways,
            "Waterways",
            "#00A6D6",
            weight=1.2,
            fill_opacity=0.0,
            show=True,
        )

    if show_aoi:

        add_vector_layer(
            fmap,
            aoi,
            "Study area boundary",
            COLOURS["dark"],
            weight=2,
            fill_opacity=0.0,
            show=True,
        )

    # --------------------------------------------------------
    # FIT MAP TO THE COMPLETE STUDY AREA
    # --------------------------------------------------------

    aoi_bounds = aoi.total_bounds

    fmap.fit_bounds(
        [
            [aoi_bounds[1], aoi_bounds[0]],
            [aoi_bounds[3], aoi_bounds[2]],
        ],
        padding=(28, 28),
        max_zoom=16,
    )

    # --------------------------------------------------------
    # MAP CONTROLS
    # --------------------------------------------------------

    Fullscreen(
        position="topleft",
        title="Full screen",
        title_cancel="Exit full screen",
        force_separate_button=True,
    ).add_to(fmap)

    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Coordinates:",
        num_digits=6,
    ).add_to(fmap)

    LayerControl(
        position="topright",
        collapsed=True,
    ).add_to(fmap)

    # --------------------------------------------------------
    # DISPLAY FULL-WIDTH MAP
    # --------------------------------------------------------

    st_folium(
        fmap,
        use_container_width=True,
        height=map_height,
        returned_objects=[],
        key="project_map",
    )

    # --------------------------------------------------------
    # DYNAMIC THEMATIC LEGEND BELOW THE MAP
    # --------------------------------------------------------

    st.divider()

    render_selected_map_legend(
        show_integrated=show_integrated,
        show_heat=show_heat,
        show_stormwater=show_stormwater,
        show_waterway_suitability=show_waterway_suitability,
        show_interventions=show_interventions,
    )

    st.caption(
        "Wetlands, waterways and the study-area boundary remain "
        "selectable reference overlays. The thematic legend displays "
        "only the selected priority and intervention layers."
    )

    st.warning(
        "Displaying several priority rasters together can cause "
        "overlap. Reduce raster opacity in the sidebar if the "
        "underlying basemap or other features become difficult to see."
    )


# ============================================================
# TAB 3: STATISTICS
# ============================================================

with tab_stats:

    st.header("Spatial Data Statistics")

    st.markdown(
        """
        These statistics describe the available input data. Raster
        means and ranges are calculated from valid pixels, excluding
        masked or NoData cells.
        """
    )

    raster_rows = []

    raster_keys = [
        ("Integrated blue-green priority", integrated_info),
        ("Heat-reduction priority", heat_info),
        ("Stormwater priority", storm_info),
        ("Intervention types", intervention_info),
        ("Waterway proximity suitability", waterway_info),
    ]

    for label, info in raster_keys:

        stats = info["stats"]

        raster_rows.append(
            {
                "Raster": label,
                "Width (pixels)": info["width"],
                "Height (pixels)": info["height"],
                "Valid pixels": stats["valid_pixels"],
                "Minimum": stats["minimum"],
                "Maximum": stats["maximum"],
                "Mean": stats["mean"],
                "CRS": info["crs"],
            }
        )

    raster_df = pd.DataFrame(raster_rows)

    st.subheader("Raster summary")

    display_raster_df = raster_df.copy()

    for column in ["Minimum", "Maximum", "Mean"]:

        display_raster_df[column] = display_raster_df[column].apply(
            lambda x: format_number(x, 3)
        )

    st.dataframe(
        display_raster_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download raster statistics (CSV)",
        data=raster_df.to_csv(index=False).encode("utf-8"),
        file_name="cooling_city_raster_statistics.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("Vector layer summary")

    vector_df = pd.DataFrame(
        [
            calculate_vector_summary(aoi, "Study area"),
            calculate_vector_summary(waterways, "Waterways"),
            calculate_vector_summary(wetlands, "Wetlands"),
            calculate_vector_summary(
                intervention_polygons,
                "Blue-green intervention polygons",
            ),
        ]
    )

    st.dataframe(
        vector_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download vector summary (CSV)",
        data=vector_df.to_csv(index=False).encode("utf-8"),
        file_name="cooling_city_vector_summary.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("Integrated priority score distribution")

    try:

        integrated_array, _, _, _, _, _, _ = read_raster_array(
            str(file_path("integrated"))
        )

        valid_values = integrated_array[
            np.isfinite(integrated_array)
        ]

        if valid_values.size > 0:

            distribution = (
                pd.Series(valid_values, name="Score")
                .round(1)
                .value_counts()
                .sort_index()
                .rename_axis("Score")
                .to_frame("Pixels")
            )

            st.bar_chart(distribution)

        else:

            st.info("The integrated raster has no valid pixels.")

    except Exception as exc:

        st.warning(
            "The priority distribution could not be calculated."
        )
        st.exception(exc)

    st.caption(
        "Pixel counts are not the same as area unless raster resolution "
        "and valid-pixel coverage are taken into account."
    )


# ============================================================
# TAB 4: SCENARIO LAB
# ============================================================

def apply_scenario_preset():
    """Update scenario sliders when a preset is selected."""

    presets = {
        "Balanced planning": {
            "heat_weight": 34,
            "storm_weight": 33,
            "ecology_weight": 33,
        },
        "Heat reduction priority": {
            "heat_weight": 60,
            "storm_weight": 25,
            "ecology_weight": 15,
        },
        "Stormwater management priority": {
            "heat_weight": 20,
            "storm_weight": 60,
            "ecology_weight": 20,
        },
        "Ecological conservation priority": {
            "heat_weight": 20,
            "storm_weight": 20,
            "ecology_weight": 60,
        },
    }

    selected = st.session_state["scenario_preset"]

    if selected in presets:
        for key, value in presets[selected].items():
            st.session_state[key] = value


def normalise_weights(heat, storm, ecology):

    total = heat + storm + ecology

    if total == 0:
        return {
            "Heat reduction": 0.0,
            "Stormwater management": 0.0,
            "Ecological conservation": 0.0,
        }

    return {
        "Heat reduction": heat / total * 100,
        "Stormwater management": storm / total * 100,
        "Ecological conservation": ecology / total * 100,
    }


def scenario_description(weights):

    highest = max(weights.values())

    dominant = [
        name
        for name, value in weights.items()
        if value == highest
    ]

    if len(dominant) == 3:
        return (
            "This scenario gives equal importance to heat reduction, "
            "stormwater management and ecological conservation."
        )

    if len(dominant) > 1:
        return (
            "This scenario gives joint highest importance to "
            + " and ".join(dominant)
            + "."
        )

    objective = dominant[0]

    descriptions = {
        "Heat reduction": (
            "The scenario places the greatest emphasis on urban cooling. "
            "Planning attention could focus on tree corridors, vegetation "
            "retention and other cooling opportunities."
        ),
        "Stormwater management": (
            "The scenario places the greatest emphasis on stormwater "
            "management. Planning attention could focus on infiltration "
            "areas, rain gardens, retention spaces and vegetated drainage."
        ),
        "Ecological conservation": (
            "The scenario places the greatest emphasis on ecological "
            "conservation. Planning attention could focus on wetlands, "
            "riparian restoration, vegetation protection and connectivity."
        ),
    }

    return descriptions[objective]


with tab_scenario:

    st.header("Scenario Lab")

    st.markdown(
        """
        Explore alternative planning priorities by changing the relative
        importance assigned to heat reduction, stormwater management
        and ecological conservation.

        The purpose of this tool is to examine how different planning
        priorities influence the decision-making framework.
        """
    )

    st.info(
        "The Scenario Lab is a conceptual weighting tool. It does not "
        "recalculate the source rasters, create new spatial priority "
        "surfaces or identify new intervention locations."
    )

    # --------------------------------------------------------
    # SCENARIO PRESETS
    # --------------------------------------------------------

    st.subheader("1. Choose a planning scenario")

    st.selectbox(
        "Planning scenario preset",
        [
            "Balanced planning",
            "Heat reduction priority",
            "Stormwater management priority",
            "Ecological conservation priority",
            "Custom scenario",
        ],
        key="scenario_preset",
        on_change=apply_scenario_preset,
        help=(
            "Select a preset to load suggested weights, or choose "
            "Custom scenario to create your own weighting."
        ),
    )

    preset_descriptions = {
        "Balanced planning": (
            "A balanced approach gives similar importance to all "
            "three planning objectives."
        ),
        "Heat reduction priority": (
            "A heat-focused approach gives greater importance to "
            "urban cooling and vegetation."
        ),
        "Stormwater management priority": (
            "A stormwater-focused approach gives greater importance "
            "to runoff management and water-sensitive interventions."
        ),
        "Ecological conservation priority": (
            "An ecology-focused approach gives greater importance "
            "to wetland protection, riparian restoration and connectivity."
        ),
        "Custom scenario": (
            "Set your own relative importance for the three objectives."
        ),
    }

    st.caption(
        preset_descriptions[st.session_state["scenario_preset"]]
    )

    st.divider()

    # --------------------------------------------------------
    # WEIGHT ADJUSTMENT
    # --------------------------------------------------------

    st.subheader("2. Adjust planning weights")

    if "heat_weight" not in st.session_state:
        st.session_state["heat_weight"] = 34

    if "storm_weight" not in st.session_state:
        st.session_state["storm_weight"] = 33

    if "ecology_weight" not in st.session_state:
        st.session_state["ecology_weight"] = 33

    weight_col1, weight_col2, weight_col3 = st.columns(3)

    with weight_col1:

        st.markdown("**Heat reduction**")

        heat_weight = st.slider(
            "Heat reduction weight",
            min_value=0,
            max_value=100,
            step=5,
            key="heat_weight",
            label_visibility="collapsed",
        )

        st.caption(
            "Tree cover, cooling corridors and vegetation retention."
        )

    with weight_col2:

        st.markdown("**Stormwater management**")

        storm_weight = st.slider(
            "Stormwater management weight",
            min_value=0,
            max_value=100,
            step=5,
            key="storm_weight",
            label_visibility="collapsed",
        )

        st.caption(
            "Infiltration, retention, rain gardens and drainage."
        )

    with weight_col3:

        st.markdown("**Ecological conservation**")

        ecology_weight = st.slider(
            "Ecological conservation weight",
            min_value=0,
            max_value=100,
            step=5,
            key="ecology_weight",
            label_visibility="collapsed",
        )

        st.caption(
            "Wetlands, riparian protection and green connectivity."
        )

    raw_total = heat_weight + storm_weight + ecology_weight

    st.metric("Total input weight", f"{raw_total}%")

    if raw_total == 0:

        st.error(
            "All weights are zero. Increase at least one objective "
            "to calculate the relative weighting."
        )

    else:

        weights = normalise_weights(
            heat_weight,
            storm_weight,
            ecology_weight,
        )

        weights_df = pd.DataFrame(
            {
                "Planning objective": list(weights.keys()),
                "Input weight": [
                    heat_weight,
                    storm_weight,
                    ecology_weight,
                ],
                "Normalised weight (%)": [
                    round(value, 1)
                    for value in weights.values()
                ],
            }
        )

        st.divider()

        # ----------------------------------------------------
        # NORMALISED WEIGHTS
        # ----------------------------------------------------

        st.subheader("3. Normalised planning weights")

        st.markdown(
            """
            The input weights are automatically converted into
            percentages that sum to 100%. This allows scenarios with
            different total input weights to be compared consistently.
            """
        )

        normalised_col, chart_col = st.columns([1, 1.3])

        with normalised_col:

            st.dataframe(
                weights_df,
                use_container_width=True,
                hide_index=True,
            )

        with chart_col:

            chart_df = (
                weights_df.set_index("Planning objective")
                [["Normalised weight (%)"]]
            )

            st.bar_chart(chart_df)

        st.divider()

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        st.subheader("4. Scenario interpretation")

        st.write(scenario_description(weights))

        highest_weight = max(weights.values())

        dominant_objectives = [
            name
            for name, value in weights.items()
            if value == highest_weight
        ]

        st.markdown("**Highest-weighted objective:**")

        for objective in dominant_objectives:
            st.markdown(f"- {objective}")

        st.markdown("**Potential planning emphasis:**")

        if weights["Heat reduction"] == highest_weight:

            st.markdown(
                """
                - Examine opportunities for tree planting and cooling
                  corridors.
                - Consider retaining existing vegetation.
                - Check whether proposed green spaces connect with
                  existing vegetation.
                """
            )

        if weights["Stormwater management"] == highest_weight:

            st.markdown(
                """
                - Examine candidate infiltration and rain-garden locations.
                - Consider retention and detention opportunities.
                - Assess vegetated drainage and the connection to
                  existing waterways.
                """
            )

        if weights["Ecological conservation"] == highest_weight:

            st.markdown(
                """
                - Examine wetland and riparian protection opportunities.
                - Consider restoration of degraded ecological areas.
                - Identify opportunities to connect fragmented green spaces.
                """
            )

        st.divider()

        # ----------------------------------------------------
        # SCENARIO COMPARISON
        # ----------------------------------------------------

        st.subheader("5. Compare alternative planning scenarios")

        comparison_scenarios = {
            "Balanced planning": [34, 33, 33],
            "Heat reduction priority": [60, 25, 15],
            "Stormwater management priority": [20, 60, 20],
            "Ecological conservation priority": [20, 20, 60],
        }

        comparison_rows = []

        for scenario_name, values in comparison_scenarios.items():

            scenario_weights = normalise_weights(*values)

            comparison_rows.append(
                {
                    "Scenario": scenario_name,
                    "Heat (%)": round(
                        scenario_weights["Heat reduction"], 1
                    ),
                    "Stormwater (%)": round(
                        scenario_weights["Stormwater management"], 1
                    ),
                    "Ecology (%)": round(
                        scenario_weights["Ecological conservation"], 1
                    ),
                }
            )

        comparison_df = pd.DataFrame(comparison_rows)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "The comparison scenarios are illustrative weighting "
            "examples. They are not measured or modelled environmental "
            "outcomes."
        )

        st.divider()

        # ----------------------------------------------------
        # SAVE / DOWNLOAD SCENARIO
        # ----------------------------------------------------

        st.subheader("6. Export the selected scenario")

        scenario_export = pd.DataFrame(
            [
                {
                    "Scenario": st.session_state["scenario_preset"],
                    "Heat reduction input weight": heat_weight,
                    "Stormwater input weight": storm_weight,
                    "Ecological conservation input weight": ecology_weight,
                    "Heat reduction normalised (%)": round(
                        weights["Heat reduction"], 2
                    ),
                    "Stormwater normalised (%)": round(
                        weights["Stormwater management"], 2
                    ),
                    "Ecological conservation normalised (%)": round(
                        weights["Ecological conservation"], 2
                    ),
                    "Total input weight": raw_total,
                    "Interpretation": scenario_description(weights),
                }
            ]
        )

        st.download_button(
            "Download selected scenario (CSV)",
            data=scenario_export.to_csv(index=False).encode("utf-8"),
            file_name="cooling_city_selected_scenario.csv",
            mime="text/csv",
        )

    st.divider()

    # --------------------------------------------------------
    # INTERVENTION OPTIONS
    # --------------------------------------------------------

    st.subheader("7. Blue-green intervention options")

    intervention_table = pd.DataFrame(
        [
            {
                "Intervention": "Tree and cooling corridors",
                "Main objective": "Heat reduction",
                "Site checks": "Existing canopy, space and utilities",
            },
            {
                "Intervention": "Rain gardens and infiltration areas",
                "Main objective": "Stormwater management",
                "Site checks": "Soil infiltration, groundwater and drainage",
            },
            {
                "Intervention": "Retention and detention spaces",
                "Main objective": "Stormwater management",
                "Site checks": "Storage capacity, inflow and overflow",
            },
            {
                "Intervention": "Wetland and riparian restoration",
                "Main objective": "Ecological conservation",
                "Site checks": "Wetland boundary, hydrology and land status",
            },
            {
                "Intervention": "Connected green corridors",
                "Main objective": "Ecological connectivity",
                "Site checks": "Habitat links, crossings and land use",
            },
            {
                "Intervention": "Vegetated drainage",
                "Main objective": "Combined benefits",
                "Site checks": "Gradient, flow capacity and maintenance",
            },
        ]
    )

    st.dataframe(
        intervention_table,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The scenario weights are planning preferences. To generate "
        "actual alternative priority maps, the underlying raster "
        "indicators would need to be combined using a documented "
        "weighted-overlay method, with consistent spatial alignment "
        "and appropriate sensitivity testing."
    )


# ============================================================
# TAB 5: METHODS & LIMITATIONS
# ============================================================

with tab_methods:

    st.header("Methods & Limitations")

    st.subheader("Analytical approach")

    st.markdown(
        """
        The project uses a GIS-based spatial decision-support approach
        to identify candidate locations for an integrated blue-green
        infrastructure network.

        The analytical framework draws on mapped indicators related to
        heat reduction, stormwater management, vegetation and ecological
        priorities. The integrated priority surface supports exploration
        of locations where several planning objectives may overlap.
        """
    )

    st.subheader("Main data inputs")

    methods_df = pd.DataFrame(
        [
            {
                "Input": "Study area boundary",
                "Purpose": "Defines the area of analysis",
            },
            {
                "Input": "Integrated blue-green priority raster",
                "Purpose": "Displays combined spatial priority",
            },
            {
                "Input": "Heat-reduction priority raster",
                "Purpose": "Explores relative heat-related priority",
            },
            {
                "Input": "Stormwater priority raster",
                "Purpose": "Explores relative stormwater-related priority",
            },
            {
                "Input": "Waterway proximity suitability raster",
                "Purpose": "Supports waterway-related suitability analysis",
            },
            {
                "Input": "Blue-green intervention polygons",
                "Purpose": "Displays candidate intervention categories",
            },
            {
                "Input": "Waterways vector layer",
                "Purpose": "Provides mapped waterway context",
            },
            {
                "Input": "Wetlands vector layer",
                "Purpose": "Provides mapped wetland context",
            },
        ]
    )

    st.dataframe(
        methods_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("How to interpret the dashboard")

    st.markdown(
        """
        1. **Study Area Overview:** review the project objectives and
           available indicators.
        2. **Map Explorer:** select one or more layers and compare them
           visually with waterways, wetlands and the study boundary.
        3. **Statistics:** inspect raster ranges, valid pixels, means
           and vector feature counts.
        4. **Scenario Lab:** compare conceptual weighting preferences
           and export a selected scenario.
        5. **Methods & Limitations:** review the intended use and
           constraints of the spatial outputs.
        """
    )

    st.subheader("Important limitations")

    st.markdown(
        """
        - **Priority is not proof of performance.** A high-priority
          pixel does not guarantee reduced temperature or flooding.
        - **Raster values depend on processing.** Interpretation must
          follow the source data, classification rules, resolution
          and NoData treatment.
        - **Spatial alignment matters.** Raster layers should use
          consistent coordinate systems, resolution, grid alignment
          and valid coverage before they are combined.
        - **Vector data may be incomplete.** Mapped waterways and
          wetlands may not capture every feature on the ground.
        - **Field verification is necessary.** Soil conditions,
          drainage pathways, infrastructure, land access and local
          constraints require further investigation.
        - **Scenario weights are illustrative.** The Scenario Lab
          does not recalculate spatial suitability surfaces.
        - **This is not an engineering design.** Drainage capacity,
          structural safety, flood levels and construction requirements
          need appropriate technical assessment.
        """
    )

    st.info(
        "Use the dashboard as a spatial screening and communication tool. "
        "Final recommendations should be supported by verified data, "
        "documented analytical methods and appropriate field assessment."
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "Cooling the City, Holding the Rains | GIS-based blue-green "
    "infrastructure planning | Exploratory decision-support dashboard"
)
