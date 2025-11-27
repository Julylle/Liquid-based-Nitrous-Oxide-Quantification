from __future__ import annotations

"""
liquidbased_quantification
--------------------------
Class-based implementation for liquid-based quantification of kLa and N₂O flux.

This version encapsulates all steps (IQR filtering, interpolation,
kLa estimation, and flux computation) within a single `LiquidQuantifier`
class, making it reusable and configurable.

Typical usage example:
    from liquidbased_quantification import LiquidQuantifier

    model = LiquidQuantifier(
        file_path="data/Monitoring_Data_Example.xlsx",
        D_R=7.55,
        AerationFieldSize=462.0
    )

    model.run_pipeline(show_plots=True)
    results = model.results
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LiquidQuantifier:
    """Encapsulates preprocessing, kLa estimation, and N₂O flux quantification.

    Attributes:
        file_path: Path to Excel data file.
        D_R: Reactor depth (m).
        AerationFieldSize: Aeration field area (m²).
        raw_data: Original raw data loaded from file.
        results: Final processed DataFrame after running the pipeline.
    """

    def __init__(self, file_path: str, D_R: float = 7.55, AerationFieldSize: float = 462.0):
        """Initializes the quantifier.

        Args:
            file_path: Path to Excel dataset.
            D_R: Reactor depth (m).
            AerationFieldSize: Aeration field area (m²).
        """
        self.file_path = file_path
        self.D_R = D_R
        self.AerationFieldSize = AerationFieldSize
        self.raw_data: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None

    # ----------------------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load Excel file into a DataFrame and set datetime index.

        Returns:
            pd.DataFrame: Loaded data.
        """
        self.raw_data = pd.read_excel(self.file_path, header=0, index_col=0)
        self.raw_data.index = pd.to_datetime(self.raw_data.index)
        return self.raw_data

    # ----------------------------------------------------------------------
    # Preprocessing and outlier removal
    # ----------------------------------------------------------------------

    @staticmethod
    def remove_outliers_iqr(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Remove outliers using the Interquartile Range (IQR) method.

        Args:
            df: DataFrame to clean.
            cols: List of columns to apply IQR filter.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        filtered = df.copy()
        for col in cols:
            if col not in filtered.columns:
                continue
            Q1, Q3 = filtered[col].quantile(0.25), filtered[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            filtered = filtered[(filtered[col] >= lower) & (filtered[col] <= upper)]
        return filtered

    def pretreat_iqr_by_group(self, df: pd.DataFrame, col: str, freq: str) -> pd.Series:
        """Apply IQR-based outlier removal grouped by frequency.

        Args:
            df: Input DataFrame indexed by datetime.
            col: Column name to clean.
            freq: Frequency string (e.g. '1H', '0.5H').

        Returns:
            pd.Series: Filtered column.
        """
        return (
            df.groupby(pd.Grouper(freq=freq))
            .apply(lambda g: self.remove_outliers_iqr(g, [col]))
            .reset_index(level=0, drop=True)[col]
        )

    # ----------------------------------------------------------------------
    # Interpolation and filling
    # ----------------------------------------------------------------------

    @staticmethod
    def fill_and_interpolate(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """Reindex to uniform time grid and interpolate linearly.

        Args:
            df: Input DataFrame.
            freq: Frequency for interpolation (default '1min').

        Returns:
            pd.DataFrame: Interpolated DataFrame.
        """
        new_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
        df = df.reindex(new_index)
        return df.interpolate(method='linear')

    # ----------------------------------------------------------------------
    # Plotting of raw VS pretereatment results
    # ----------------------------------------------------------------------
    
    @staticmethod
    def compare_plot(df: pd.DataFrame, cols: List[str], pretreat: str, ylabel: str,
                    size: Tuple[int, int] = (10, 4)) -> None:
        """Scatter comparison plot between raw and pretreated data."""
        plt.figure(figsize=size)
        
        if cols[0] in df:
            plt.scatter(
                df.index,
                df[cols[0]].to_numpy().ravel(),
                color="#f15bb5",
                marker="+",
                label=f"Before {pretreat}"
            )

        if cols[1] in df:
            plt.scatter(
                df.index,
                df[cols[1]].to_numpy().ravel(),
                color="green",
                marker=".",
                label=f"After {pretreat}"
            )

        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.show()


    # ----------------------------------------------------------------------
    # Plotting of single varibale time series
    # ----------------------------------------------------------------------
    
    @staticmethod
    def single_variable_plot(plot_data: pd.DataFrame, size: Tuple[int, int], ylabel: str) -> None:
        """Plot a single time-series variable (handles Series or single-column DataFrame)."""
        import matplotlib.pyplot as plt
        import numpy as np

        # Accept either a Series or a single-column DataFrame
        if isinstance(plot_data, pd.DataFrame):
            if plot_data.shape[1] > 1:
                raise ValueError("Expected a single-column DataFrame or Series for plotting.")
            y = np.asarray(plot_data.iloc[:, 0]).reshape(-1)
            x = np.asarray(plot_data.index)
        else:
            y = np.asarray(plot_data).reshape(-1)
            x = np.asarray(plot_data.index)

        ls = 25
        font = {"weight": "normal", "size": ls}
        w, h = size

        plt.figure(figsize=(w, h))
        plt.plot(
            x,
            y,
            color="purple",
            marker="s",
            markersize=10,
            linestyle="-",
            linewidth=3.5,
            label=ylabel,
        )
        plt.ylabel(ylabel, font)
        plt.xlabel("Date", font)
        plt.tick_params(labelsize=ls)
        plt.legend(prop=font, loc="lower right", ncol=1, frameon=True)
        plt.tight_layout()
        plt.show()

    

    # ----------------------------------------------------------------------
    # kLa and flux estimation
    # ----------------------------------------------------------------------

    def sfv_kla(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate kLa using the SFV approach.

        Args:
            df: DataFrame with interpolated airflow and temperature.

        Returns:
            pd.DataFrame: Copy with new 'kla_SFV' column.
        """
        out = df.copy()
        Qair = out['After_inter_Airflow']
        Vg = Qair / self.AerationFieldSize
        D_L = 0.815
        temp_corr = np.power(1.024, out['After_inter_LiqTemp'] - 20)
        out['kla_SFV'] = np.power(self.D_R / D_L, -0.49) * 34500 * np.power(Vg, 0.86) * temp_corr
        return out

    def flux_estimation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate N₂O flux using Henry’s Law and mass transfer equation."""
        out = df.copy()
        Vaerated = self.AerationFieldSize * self.D_R
        Qair = out['After_inter_Airflow'] * 60 * 60 * 24
        Temp = out['After_inter_LiqTemp'] + 273.15
        kla = out['kla_SFV']
        liqN2O = out['After_inter_LiqN2O']

        kH0, R, T0, solnHR = 0.0247, 8.314e-5, 25 + 273.15, 2675
        kH = kH0 * np.exp(solnHR * (1 / Temp - 1 / T0))
        H_N2O = 1 / (kH * R * Temp * 1e3)
        out['EstimatedFlux'] = H_N2O * liqN2O * (1 - np.exp(-kla / H_N2O * Vaerated / Qair)) * Qair / 24 / self.AerationFieldSize
        return out

    # ----------------------------------------------------------------------
    # Pipeline
    # ----------------------------------------------------------------------

    def run_pipeline(self, show_plots: bool = True) -> pd.DataFrame:
        """Execute the full workflow.

        Args:
            show_plots: Whether to display plots during execution.

        Returns:
            pd.DataFrame: Final DataFrame with kLa and flux results.
        """
        raw = self.load_data()

        df = pd.DataFrame(index=raw.index)
        df['RawAirflow'] = raw['Airflow_rate_(m3/s)']
        df['RawLiqN2O'] = raw['Liquid_N2O_(mgN/L)']
        df['RawLiqTemp'] = raw['Liquid_temperature_(°C)']

        # Step 1: IQR filtering
        df['After_IQR_Airflow'] = self.pretreat_iqr_by_group(raw, 'Airflow_rate_(m3/s)', '1H')
        df['After_IQR_LiqN2O'] = self.pretreat_iqr_by_group(raw, 'Liquid_N2O_(mgN/L)', '0.5H')
        df['After_IQR_LiqTemp'] = self.pretreat_iqr_by_group(raw, 'Liquid_temperature_(°C)', '1H')

        if show_plots:
            self.compare_plot(df, ['RawAirflow', 'After_IQR_Airflow'], 'IQR', 'Airflow (m$^3$/s)')
            self.compare_plot(df, ['RawLiqN2O', 'After_IQR_LiqN2O'], 'IQR', 'Liq N$_2$O (mgN/L)')
            self.compare_plot(df, ['RawLiqTemp', 'After_IQR_LiqTemp'], 'IQR', 'Temperature (°C)')

        # Step 2: Interpolation
        df = self.fill_and_interpolate(df)
        df['After_inter_Airflow'] = df['After_IQR_Airflow']
        df['After_inter_LiqN2O'] = df['After_IQR_LiqN2O']
        df['After_inter_LiqTemp'] = df['After_IQR_LiqTemp']

        # Step 3: kLa and Flux estimation
        kla_df = self.sfv_kla(df)
        flux_df = self.flux_estimation(kla_df)

        if show_plots:
            hourly_kla = flux_df[['kla_SFV']].resample('H').mean()
            hourly_flux = flux_df[['EstimatedFlux']].resample('H').mean()
            self.single_variable_plot(hourly_kla, [25, 8], 'k$_L$a$_{N_2O}$ (d$^{-1}$)')
            self.single_variable_plot(hourly_flux, [25, 8], 'N$_2$O flux (g-N/h-m$^2$)')

        self.results = flux_df
        return flux_df
