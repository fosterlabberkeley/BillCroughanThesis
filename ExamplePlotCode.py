import numpy as np
import time
import os
import seaborn as sns
import pandas as pd

from PlotUtil import PlotManager, PlotContext, violinPlot
from Shuffler import ShuffSpec


def main():
    if False:
        # ==================== Quick start ====================
        pm = PlotManager()
        with pm.newFig("fig1") as pc:
            pc.ax.plot([1, 2, 3], [1, 4, 9], label="x^2")
        print("that's it! Check your current directory for a file called 'fig1.png'")
        input("Press enter to continue...")

    # ==================== Doing more ====================
    # To avoid clutter, store output in a subdirectory
    globalOutputDir = os.path.join(os.curdir, "examplePlots")
    # Generate a random seed for reproducibility
    seed = int(time.perf_counter())
    print("random seed =", seed)

    if False:
        # Now create our PlotManager object
        pm = PlotManager.getGlobalPlotManager(
            outputDir=globalOutputDir, randomSeed=seed)

        # Note now anywhere we ask for this global PlotManager, we get the same one
        # If we instead wanted a local PlotManager, we could use the following code:
        # pm = PlotManager(outputDir=globalOutputDir, randomSeed=seed)

        # ==================== Plotting basics ====================
        # Now we can create figures
        with pm.newFig("fig1") as pc:
            # pc is a PlotContext object. pc.ax is a matplotlib Axes object, which we can plot onto
            pc.ax.plot([1, 2, 3], [1, 4, 9], label="x^2")
            pc.ax.set_title("Example Figure 1")
            # When we exit the with block, the figure is saved to disk
        print("Done with figure 1!")

        # We can continue plotting to this figure
        with pm.continueFig("fig1_continued") as pc:
            pc.ax.scatter([1, 2], [3, 4], label="x+2")
        print("Done with figure 1 continued!")

        # We can also create a new figure, which will clear all old data
        with pm.newFig("fig2") as pc:
            x = np.linspace(0, 2 * np.pi)
            pc.ax.plot(x, np.sin(x), label="sin(x)")
            pc.ax.set_title("Example Figure 2")
        print("Done with figure 2!")

        # Multiple axes
        with pm.newFig("fig3", subPlots=(2, 1)) as pc:
            x = np.linspace(0, 2 * np.pi)
            # pc.axs holds the array of axes
            pc.axs[0].plot(x, np.sin(x), label="sin(x)")
            pc.axs[1].plot(x, np.cos(x), label="cos(x)")

            # pc.ax will give you the first subplot if there are multiple
            pc.ax.set_title("First subplot")
            pc.axs[1].set_title("Second subplot")
        print("Done with figure 3!")

        # Each plot can be shown or saved, independent of the default
        with pm.newFig("nosave", showPlot=True, savePlot=False) as pc:
            pc.ax.plot([1, 2, 3], [1, 4, 9], label="x^2")
            pc.ax.set_title("This figure isn't saved")
        print("Done with figure nosave!")
        input("Press enter to continue...")

    # ==================== Plotting with multiple subsets of data ====================
    # Let's generate a dataset with multiple subjects
    numRats = 2
    rats = []
    for ri in range(numRats):
        rat = {"name": "rat" + str(ri)}
        n = np.random.randint(30, 50)
        rat["data1"] = np.random.normal(0, 1, n)
        rat["category1"] = np.random.choice(["A", "B"], n)
        rat["category2"] = np.random.choice(["X", "Y", "Z"], n)
        rat["data2"] = np.random.normal(ri, 1, n) ** 2 + \
            np.where(rat["category2"] == "X", 0, 1) + \
            np.where(rat["category1"] == "A", 0, 2) + \
            np.where(rat["category2"] == "X", 0, 1) * \
            np.where(rat["category1"] == "A", 0, 1) * 0.5
        rats.append(rat)

    if False:
        for rat in rats:
            # Figures will be saved to a new subdirectory for each rat
            pm.setOutputSubDir(rat["name"])
            with pm.newFig("ratData") as pc:
                pc.ax.hist(rat["data1"], label="data1")
        pm.setOutputSubDir("")  # Reset the output subdirectory

        # Now we can make a combined figure
        pm.makeCombinedFigs(alignAxes="x")
        print("Done with combined figure!")
        input("Press enter to continue...")

        # ==================== Now let's add in the stats ====================
        for rat in rats:
            # This tells the shuffler all the following data is from the same rat
            pm.setStatCategory("rat", rat["name"])
            # We can also use a push-pop system to set the output directories
            pm.pushOutputSubDir(rat["name"])

            with pm.newFig("data1") as pc:
                violinPlot(pc.ax, rat["data1"], rat["category1"])
                # Specify what values we want to shuffle
                pc.yvals["data1"] = rat["data1"]
                pc.categories["category1"] = rat["category1"]
                # And specify the shuffle to perform. We can also omit this and wait until later
                # But specifying it now will also add a graphic to the plot itself
                numShuffles = 100
                pc.immediateShuffles.append((
                    [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="category1", value="A")], numShuffles))

            with pm.newFig("data2") as pc:
                violinPlot(pc.ax, rat["data2"], rat["category2"])
                # Specify what values we want to shuffle
                pc.yvals["data2"] = rat["data2"]
                pc.categories["category2"] = rat["category2"]
                # And specify the shuffle to perform. We can also omit this and wait until later
                # But specifying it now will also add a graphic to the plot itself
                numShuffles = 100
                pc.immediateShuffles.append((
                    [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="category2", value="X")], numShuffles))

            with pm.newFig("data2_bothCats") as pc:
                violinPlot(pc.ax, rat["data2"], rat["category2"],
                           categories2=rat["category1"])
                # Specify what values we want to shuffle
                pc.yvals["data2"] = rat["data2"]
                pc.categories["category2"] = rat["category2"]
                pc.categories["category1"] = rat["category1"]
                # And specify the shuffle to perform. We can also omit this and wait until later
                # But specifying it now will also add a graphic to the plot itself
                numShuffles = 100
                pc.immediateShuffles.append((
                    [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="category1", value="A")], numShuffles))

            # Just remember to pop the output directory when you're done
            pm.popOutputSubDir()

        pm.makeCombinedFigs()
        pm.runImmediateShufflesAcrossPersistentCategories()
        print("Done with combined figure with stats!")
        input("Press enter to continue...")

        # ==================== Other use cases ====================
        # You can also run all possible shuffles
        # Note keeping going after running the shuffles is totally untested
        # It might work, but to make things easy i'm going to create a new local plot manager
        pm = PlotManager(outputDir=globalOutputDir, randomSeed=seed)
        for rat in rats:
            pm.setStatCategory("rat", rat["name"])
            pm.pushOutputSubDir(rat["name"])

            with pm.newFig("data2_bothCats") as pc:
                violinPlot(pc.ax, rat["data2"], rat["category2"],
                           categories2=rat["category1"])
                # Specify what values we want to shuffle later on
                pc.yvals["data2"] = rat["data2"]
                pc.categories["category2"] = rat["category2"]
                pc.categories["category1"] = rat["category1"]

            pm.popOutputSubDir()

        pm.runShuffles()
        print("Done with combined figure with all shuffles!")
        input("Press enter to continue...")

    # Custom shuffle functions. For when you want to add some restriction on shuffles
    # Example use case is if something was pseudorandomly generated, and you want to shuffle
    # the pseudorandomness, but not the actual data
    for rat in rats:
        # Let's make some pairwise data where each pair is randomly assigned -1,1 or 1,-1
        n = len(rat["data1"])
        rat["pseudoRandomGroup"] = np.repeat(np.arange(n//2 + 1), 2)[:n]
        pseudoRandomCategories = np.where(
            np.random.normal(0, 1, n//2 + 1) > 0, 1, -1)
        rat["pseudoRandomCategories"] = np.repeat(
            pseudoRandomCategories, 2)[:n]
        rat["pseudoRandomCategories"][1::2] *= -1

    def prShuffler(df: pd.DataFrame, colName: str, rng: np.random.Generator) -> pd.Series:
        def swapFunc(val): return -1 if val == 1 else 1
        numGroups = len(df["pseudoRandomGroup"].unique())
        swapBool = (np.random.uniform(size=(numGroups,)) < 0.5).astype(bool)
        return [swapFunc(val) if swapBool[prg] else val for prg, val in zip(df["pseudoRandomGroup"], df[colName])]

    pm = PlotManager(outputDir=globalOutputDir, randomSeed=seed)
    pm.setCustomShuffleFunction("pseudoRandomCategories", prShuffler)

    with pm.newFig("customShuffle") as pc:
        violinPlot(pc.ax, rats[0]["data2"], rats[0]
                   ["pseudoRandomCategories"], categories2=rats[0]["category2"])
        pc.yvals["customShuffle"] = rats[0]["data2"]
        pc.categories["pseudoRandomCategories"] = rats[0]["pseudoRandomCategories"]
        pc.categories["category2"] = rats[0]["category2"]
        pc.immediateShuffles.append(
            ([ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="pseudoRandomCategories")], 100))

    print("Done with custom shuffle!")
    input("Press enter to continue...")

    # If multiple figures use the same filename but should not be combined, can use uniqueID
    pm = PlotManager(outputDir=globalOutputDir, randomSeed=seed)
    for rat in rats:
        pm.setStatCategory("rat", rat["name"])
        pm.pushOutputSubDir(rat["name"])
        n = len(rat["data1"])

        pm.pushOutputSubDir("firstHalf")

        with pm.newFig("data1", uniqueID="data1_firsthalf") as pc:
            violinPlot(pc.ax, rat["data1"][:n//2], rat["category1"][:n//2])
            pc.yvals["data1"] = rat["data1"][:n//2]
            pc.categories["category1"] = rat["category1"][:n//2]

        pm.popOutputSubDir()
        pm.pushOutputSubDir("secondHalf")

        with pm.newFig("data1", uniqueID="data1_secondhalf") as pc:
            violinPlot(pc.ax, rat["data1"][n//2:], rat["category1"][n//2:])
            pc.yvals["data1"] = rat["data1"][n//2:]
            pc.categories["category1"] = rat["category1"][n//2:]

        pm.popOutputSubDir()

    pm.makeCombinedFigs()
    print("Done with uniqueID!")
    print("Goodbye!")


if __name__ == "__main__":
    main()
