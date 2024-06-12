import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os
import random

light_colors = ["whitesmoke", "white", "snow", "mistyrose", "seashell", "linen",
                "antiquewhite", "oldlace", "floralwhite", "cornsilk", "lemonchiffon",
                "ivory", "beige", "lightyellow", "lightgoldenrodyellow",
                "yellow", "honeydew", "mintcream", "azure", "lightcyan", "aliceblue",
                "ghostwhite", "lavender", "lavenderblush"]
all_colors = list(mcolors.CSS4_COLORS)
COLOR_LIST = [color for color in all_colors if color not in light_colors]

class PlotData():
    def __init__(self) -> None:
        pass
        
    def just_plot(self, folder_path: str):
        # compiled_csv_list = []
        # find all the compiled.csv in the given folder
        for folders, dirs, fnames in os.walk(folder_path):
            for fname in fnames:
                if fname.endswith("_compiled.csv") and "._" not in fname:
                    file_path = os.path.join(folders, fname)
                    df = self._read_csv(file_path)
                    save_path = file_path[:-len("_compiled.csv")]
                    rec_name = df.loc[:, 'name']
                    genotypes, groups, _ = self._group_data(rec_name)

                    cols = df.columns
                    for col in cols:
                        if col == "name" or "Standard Deviation" in col:
                            continue
                        self._plot(genotypes, groups, df, col, save_path, None, "")

                elif fname.endswith("_compiled_st.csv") and "._" not in fname:
                    file_path = os.path.join(folders, fname)
                    df = self._read_csv(file_path)
                    save_path = file_path[:-len("_compiled_st.csv")] + "_ST"
                    rec_name = df.loc[:, 'name']
                    genotypes, groups, _ = self._group_data(rec_name)

                    cols = df.columns
                    for col in cols:
                        if col == "name" or "Standard Deviation" in col:
                            continue
                        self._plot(genotypes, groups, df, col, save_path, "ST", "")

                elif fname.endswith("_compiled_nst.csv") and "._" not in fname:
                    file_path = os.path.join(folders, fname)
                    df = self._read_csv(file_path)
                    save_path = file_path[:-len("_compiled_nst.csv")] + "_NST"
                    rec_name = df.loc[:, 'name']
                    genotypes, groups, _ = self._group_data(rec_name)

                    cols = df.columns
                    for col in cols:
                        if col == "name" or "Standard Deviation" in col:
                            continue
                        self._plot(genotypes, groups, df, col, save_path, "NST", "")

    def ana_plot(self, csv_path: str, evk: str | None, recording_group: str):
        """Plot after analysis."""
        df = self._read_csv(csv_path)

        if evk == "ST":
            save_path = csv_path[:-len("_compiled_st.csv")] + "ST"
        elif evk == "NST":
            save_path = csv_path[:-len("_compiled_nst.csv")] + "NST"
        else:
            save_path = csv_path[:-len("_compiled.csv")]

        rec_name = df.loc[:, 'name']
        genotypes, groups, _ = self._group_data(rec_name)
        # print(f"================Groups are {groups}")

        cols = df.columns
        for col in cols:
            if col == "name" or "Standard Deviation" in col:
                continue
            self._plot(genotypes, groups, df, col, save_path, None, recording_group)

    def _read_csv(self, path: str) -> pd.DataFrame:
        """Read the csv file."""
        with open(path, "r") as file:
            dff_file = pd.read_csv(file)
        return dff_file

    def _group_data(self, fnames: list[str]):
        """Group the names."""
        genotypes={}
        groups = {}
        diff = []
        first_group = ""

        first_name = fnames[0].split('_')     
        for i, name in enumerate(fnames):
            elements = name.split('_')
            if i == 0:
                first_group = elements.index("MMStack") + 1

            genotype = self._genotype(elements)
            if not genotypes.get(genotype):
                genotypes[genotype] = []
            genotypes[genotype].append(genotype)

            diff_ele = [ele for ele in elements if ele not in first_name and\
                         len(ele)>1 and not ele.startswith("Pos")]
            if len(diff_ele) == 0:
                if not groups.get(elements[first_group]):
                    groups[elements[first_group]] = []
                    diff.append(elements[first_group])
                groups[elements[first_group]].append(i)
            elif len(diff_ele) == 1:
                if (ele for ele in diff) in diff_ele:
                    continue
                if not groups.get(diff_ele[0]):
                    groups[diff_ele[0]] = []
                    diff.append(diff_ele)
                groups[diff_ele[0]].append(i)
                
        
        return genotypes, groups, diff

    def _genotype(self, element_list: list[str]) -> str:
        """Define genotype."""
        neg = element_list.count('-')
        pos = element_list.count('+')

        if neg == 1 and pos == 1:
            return "het"
        elif neg == 2 and pos == 0:
            return "null"
        elif neg == 0 and pos == 2:
            return "control"

    def _get_data(self, all_data: pd.DataFrame, metric: str, group_ind: list) -> list:
        """Get data for one group."""
        group_data = []
        for ind in group_ind:
            group_data.append(all_data.loc[ind, metric])
        
        return group_data

    def _plot(self, genotypes: dict, groups: dict, all_data: pd.DataFrame, 
              metric: str, path: str, evk: str | None, recording_group: str):
        """Plot the metric """
        fig, ax = plt.subplots()
        start_x = 1

        for geno in genotypes:
            for group, index in groups.items():
                data = self._get_data(all_data, metric, index)

                x_range = np.ones(len(data)) * start_x
                ax.scatter(x_range, data, label=group)

                title = f"{geno}_{metric}_{recording_group}"
                if evk:
                    title += f"_{evk}"
                ax.set_title(title)

                ax.set_xticks([])
                ax.legend()
                start_x += 1

        if "Average Frequency" in metric:
            metric = "Average Frequency"

        folder_path = os.path.join(path, f"{path}_{geno}_graphs")
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        save_path = os.path.join(folder_path, f"{recording_group}_{metric}.png")
        plt.savefig(save_path)
        plt.close()

    def _plot_traces(self, roi_dff: dict, spk_times: dict, save_path: str, normalized: str) -> None:
        """Plot  traces."""
        dff_to_plot, color_list = self._random_pick(roi_dff, 10)
        self._plot_traces_no_peaks(roi_dff, dff_to_plot, color_list, save_path, normalized)
        self._plot_traces_w_peaks(roi_dff, dff_to_plot, spk_times, color_list, save_path, normalized)

    def _plot_traces_no_peaks(self, roi_dff: dict, dff_to_plot: list,
                              color_list: list, path: str, normalized: str) -> None:
        """Plot the traces."""
        fig, ax = plt.subplots(figsize=(20, 20))
        if len(dff_to_plot) > 0:
            dff_max = np.zeros(len(dff_to_plot))
            for max_index, dff_index in enumerate(dff_to_plot):
                dff_max[max_index] = np.max(roi_dff[dff_index])
            height_increment = max(dff_max)

            y_pos = []
            for height_index, d in enumerate(dff_to_plot):
                y_pos.append(height_index * (1.2 * height_increment))
                ax.plot(roi_dff[d] + height_index * (1.2 * height_increment), color=color_list[height_index],
                        linewidth=3)
            ax.set_yticks(y_pos, labels=dff_to_plot)
            fname = normalized + "_traces_no_detection.png"
            plt.savefig(os.path.join(path, fname))
            plt.close()

    def _plot_traces_w_peaks(self, roi_dff: dict, dff_to_plot: list, spk_times: dict, 
                             color_list: list, path: str, normalized: str):
        """Plot traces with peak detected."""
        fig, ax = plt.subplots(figsize=(20, 20))
        if len(dff_to_plot) > 0:
            dff_max = np.zeros(len(dff_to_plot))
            for max_index, dff_index in enumerate(dff_to_plot):
                dff_max[max_index] = np.max(roi_dff[dff_index])
            height_increment = max(dff_max)

            y_pos = []
            for height_index, d in enumerate(dff_to_plot):
                y_pos.append(height_index * (1.2 * height_increment))
                ax.plot(roi_dff[d] + height_index * (1.2 * height_increment), color=color_list[height_index], linewidth=3)
                if len(spk_times[d]) > 0:
                    y = [roi_dff[d][i] for i in spk_times[d]]
                    ax.plot(spk_times[d], y + height_index * (1.2 * height_increment),
                                   ms=6, color='r', marker='o', ls='', label=f"{d}: {len(spk_times[d])}")
                    ax.legend()

            ax.set_yticks(y_pos, labels=dff_to_plot)
            fname = normalized + "_traces_w_peaks.png"
            plt.savefig(os.path.join(path, fname))
            plt.close()

    def _random_pick(self, roi_dff: dict, num: int) -> tuple[list, list]:
        """Pick 10 traces randomly to plot."""
        num_f = np.min([num, len(roi_dff)])
        final_dff = random.sample(list(roi_dff.keys()), num_f)
        final_dff.sort()
        rand_color_ind = random.sample(COLOR_LIST, k=num_f,)

        return final_dff, rand_color_ind
    