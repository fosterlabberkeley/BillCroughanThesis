from BTData import BTData
from BTSession import BTSession
import numpy as np
import os
import csv
import glob
from scipy import stats, signal
import MountainViewIO
from datetime import datetime, date
import sys
from PyQt5.QtWidgets import QApplication
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata
import dataclasses
from numpy.typing import ArrayLike

from consts import allWellNames, TRODES_SAMPLING_RATE, LFP_SAMPLING_RATE, CM_PER_FT
from UtilFunctions import readWellCoordsFile, readRawPositionData, getListOfVisitedWells, getRipplePower, \
    detectRipples, getUSBVideoFile, quickPosPlot, parseCmdLineAnimalNames, \
    timeStrForTrodesTimestamp, quadrantOfWell, TimeThisFunction, getDrivePathByLabel, getActivelinkLogFile, \
    getLoadInfo, LoadInfo, generateFoundWells, ImportOptions, parseCmdLineImportOptions, posOnWall, \
    getWellPosCoordinates
from ClipsMaker import AnnotatorWindow
from TrodesCameraExtrator import getTrodesLightTimes, processRawTrodesVideo, processUSBVideoData

importParentApp = QApplication(sys.argv)


# Returns a list of directories that correspond to runs for analysis. Unless runJustSpecified is True,
# only ever filters by day. Thus index within day of returned list can be used to find corresponding behavior notes file
def getSessionDirs(loadInfo: LoadInfo, importOptions: ImportOptions) -> Tuple[List[str], List[str]]:
    filtered_data_dirs = []
    prevSessionDirs = []
    prevSession = None
    all_data_dirs = sorted([d for d in os.listdir(loadInfo.data_dir) if d.count("_") > 0],
                           key=lambda s: (s.split('_')[0], s.split('_')[1]))

    for session_idx, session_dir in enumerate(all_data_dirs):
        if session_dir == "behavior_notes" or "numpy_objs" in session_dir:
            continue

        if not os.path.isdir(os.path.join(loadInfo.data_dir, session_dir)):
            continue

        dir_split = session_dir.split('_')
        if dir_split[-1] == "Probe" or dir_split[-1] == "ITI":
            # will deal with this in another loop
            print(f"skipping {session_dir}, is probe or iti")
            continue

        dateStr = dir_split[0][-8:]

        if importOptions.runJustSpecified and dateStr not in importOptions.specifiedDays and \
                session_dir not in importOptions.specifiedRuns:
            prevSession = session_dir
            continue

        if dateStr in loadInfo.excluded_dates:
            print(f"skipping, excluded date {dateStr}")
            prevSession = session_dir
            continue

        if loadInfo.minimum_date is not None and dateStr < loadInfo.minimum_date:
            print("skipping date {}, is before minimum date {}".format(
                dateStr, loadInfo.minimum_date))
            prevSession = session_dir
            continue

        if loadInfo.maximum_date is not None and dateStr > loadInfo.maximum_date:
            print("skipping date {}, is after maximum date {}".format(
                dateStr, loadInfo.maximum_date))
            prevSession = session_dir
            continue

        filtered_data_dirs.append(session_dir)
        if prevSession is not None:
            d1 = date(int(dateStr[0:4]), int(dateStr[4:6]), int(dateStr[6:8]))
            pds = prevSession.split('_')[0][-8:]
            d2 = date(int(pds[0:4]), int(pds[4:6]), int(pds[6:8]))
            daysBetweenSessions = (d1 - d2).days
            # print(d1, d2, daysBetweenSessions)
            if daysBetweenSessions > 2:
                prevSession = None
        prevSessionDirs.append(prevSession)
        prevSession = session_dir

    return filtered_data_dirs, prevSessionDirs


def makeSessionObj(seshDir: str, prevSeshDir: str, sessionNumber: int, prevSessionNumber: int,
                   loadInfo: LoadInfo, skipActivelinkLog=False) -> BTSession:
    sesh = BTSession()
    sesh.prevSessionDir = prevSeshDir

    dir_split = seshDir.split('_')
    dateStr = dir_split[0][-8:]
    timeStr = dir_split[1]
    session_name_pfx = dir_split[0][0:-8]
    sesh.dateStr = dateStr
    sesh.name = seshDir
    sesh.timeStr = timeStr
    s = "{}_{}".format(dateStr, timeStr)
    sesh.date = datetime.strptime(s, "%Y%m%d_%H%M%S")

    # check for files that belong to this date
    sesh.btDir = seshDir
    gl = loadInfo.data_dir + session_name_pfx + dateStr + "*"
    dir_list = glob.glob(gl)
    for d in dir_list:
        if d.split('_')[-1] == "ITI":
            sesh.itiDir = d
            sesh.separateItiFile = True
            sesh.recordedIti = True
        elif d.split('_')[-1] == "Probe":
            sesh.probeDir = d
            sesh.separateProbeFile = True
    if not sesh.separateProbeFile:
        sesh.recordedIti = True
        sesh.probeDir = os.path.join(loadInfo.data_dir, seshDir)
    if not sesh.separateItiFile and sesh.recordedIti:
        sesh.itiDir = seshDir

    # Get behavior_notes info file name
    behaviorNotesDir = os.path.join(loadInfo.data_dir, 'behavior_notes')
    sesh.infoFileName = os.path.join(behaviorNotesDir, dateStr + ".txt")
    if not os.path.exists(sesh.infoFileName):
        # Switched numbering scheme once started doing multiple sessions a day
        sesh.infoFileName = os.path.join(
            behaviorNotesDir, f"{dateStr}_{sessionNumber}.txt")
    sesh.seshIdx = sessionNumber

    if prevSeshDir is None:
        sesh.prevInfoFileName = None
        sesh.prevSessionIdx = None
    else:
        dir_split = prevSeshDir.split('_')
        prevSeshDateStr = dir_split[0][-8:]
        sesh.prevInfoFileName = os.path.join(
            behaviorNotesDir, prevSeshDateStr + ".txt")
        if not os.path.exists(sesh.prevInfoFileName):
            # Switched numbering scheme once started doing multiple sessions a day
            sesh.prevInfoFileName = os.path.join(
                behaviorNotesDir, f"{prevSeshDateStr}_{prevSessionNumber}.txt")
        sesh.prevSessionIdx = prevSessionNumber

    if not skipActivelinkLog:
        raise Exception("Need to change this drive path")
        possibleDirectories = [os.path.join(getDrivePathByLabel(
            "WDC8"), "DesktopFiles", loadInfo.animalName)]
        # print(possibleDirectories)
        sesh.activelinkLogFileName = getActivelinkLogFile(
            sesh.name, possibleDirectories=possibleDirectories)
        sesh.hasActivelinkLog = sesh.activelinkLogFileName is not None

    sesh.fileStartString = os.path.join(loadInfo.data_dir, seshDir, seshDir)
    sesh.loadInfo = loadInfo

    return sesh


def parseActivelinkLog(sesh: BTSession) -> None:
    if not sesh.hasActivelinkLog:
        return

    with open(sesh.activelinkLogFileName, 'r') as f:
        detections = []
        lines = f.readlines()
        for line in lines:
            lineInfo, lineTxt = line.split(maxsplit=1)
            moduleName = lineInfo[14:-1]
            printTime = lineInfo[:12]

            if moduleName == "RippleAnalysis":
                if "Mean LFP" in lineTxt:
                    lc = lineTxt.split(",")
                    meanVal = float(lc[0].split()[-1])
                    stdVal = float(lc[1].split()[-1])
                    sesh.loggedStats.append([printTime, meanVal, stdVal])
                elif lineTxt.startswith("Detected ripple at "):
                    ts = int(lineTxt.split("TS:")[1].split()[0][:-1])
                    detections.append(ts)

        sesh.loggedDetections_ts = np.array(detections)
        sesh.rpowmLog = sesh.loggedStats[-1][1]
        sesh.rpowsLog = sesh.loggedStats[-1][2]


def parseInfoFiles(sesh: BTSession) -> None:
    with open(sesh.infoFileName, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lineparts = line.split(":")
            if len(lineparts) != 2:
                sesh.notes.append(line)
                continue

            fieldName = lineparts[0]
            fieldVal = lineparts[1]

            # if sesh.importOptions["justExtractData"]  and fieldName.lower() not in ["reference", "ref", "baseline", \
            # "home", \
            # "aways", "last away", "last well", "probe performed"]:
            # continue

            if fieldName.lower() == "home":
                sesh.homeWell = int(fieldVal)
            elif fieldName.lower() == "aways":
                sesh.awayWells = [int(w) for w in fieldVal.strip().split(' ')]
                # print(line)
                # print(fieldName, fieldVal)
                # print(sesh.awayWells)
            elif fieldName.lower() == "foundwells":
                sesh.foundWells = [int(w) for w in fieldVal.strip().split(' ')]
            elif fieldName.lower() == "condition":
                typeIn = fieldVal.lower()
                if 'ripple' in typeIn or 'interruption' in typeIn:
                    # sesh.isRippleInterruption = True
                    sesh.condition = BTSession.CONDITION_INTERRUPTION
                elif 'none' in typeIn:
                    # sesh.isNoInterruption = True
                    sesh.condition = BTSession.CONDITION_NO_STIM
                elif 'delay' in typeIn:
                    # sesh.isDelayedInterruption = True
                    sesh.condition = BTSession.CONDITION_DELAY
                else:
                    print("\tCouldn't recognize Condition {} in file {}".format(
                        typeIn, sesh.infoFileName))
            elif fieldName.lower() == "thresh":
                if "low" in fieldVal.lower():
                    sesh.rippleDetectionThreshold = 2.5
                elif "high" in fieldVal.lower():
                    sesh.rippleDetectionThreshold = 4.0
                elif "med" in fieldVal.lower():
                    sesh.rippleDetectionThreshold = 3.0
                elif "none" in fieldVal.lower():
                    sesh.rippleDetectionThreshold = None
                else:
                    sesh.rippleDetectionThreshold = float(
                        fieldVal)
            elif fieldName.lower() == "last away":
                if fieldVal.strip() == "None":
                    sesh.lastAwayWell = None
                else:
                    sesh.lastAwayWell = int(fieldVal)
            elif fieldName.lower() == "reference" or fieldName.lower() == "ref":
                if fieldVal.strip() == "None":
                    sesh.rippleDetectionTetrodes = None
                else:
                    sesh.rippleDetectionTetrodes = [int(fieldVal)]
            elif fieldName.lower() == "baseline":
                if fieldVal.strip() == "None":
                    sesh.rippleBaselineTetrode = None
                else:
                    sesh.rippleBaselineTetrode = int(fieldVal)
            elif fieldName.lower() == "last well":
                if fieldVal.strip() == "None":
                    sesh.foundFirstHome = False
                    sesh.endedOnHome = False
                else:
                    sesh.foundFirstHome = True
                    if 'h' in fieldVal.lower():
                        sesh.endedOnHome = True
                    elif 'a' in fieldVal.lower():
                        sesh.endedOnHome = False
                    else:
                        print("\tCouldn't recognize last well {} in file {}".format(
                            fieldVal, sesh.infoFileName))
            elif fieldName.lower() == "iti stim on":
                if 'Y' in fieldVal:
                    sesh.ItiStimOn = True
                elif 'N' in fieldVal:
                    sesh.ItiStimOn = False
                else:
                    print("\tCouldn't recognize ITI Stim condition {} in file {}".format(
                        fieldVal, sesh.infoFileName))
            elif fieldName.lower() == "probe stim on":
                if 'Y' in fieldVal:
                    sesh.probeStimOn = True
                elif 'N' in fieldVal:
                    sesh.probeStimOn = False
                else:
                    print("\tCouldn't recognize Probe Stim condition {} in file {}".format(
                        fieldVal, sesh.infoFileName))
            elif fieldName.lower() == "probe performed":
                if 'Y' in fieldVal:
                    sesh.probePerformed = True
                elif 'N' in fieldVal:
                    sesh.probePerformed = False
                    if sesh.animalName == "Martin":
                        raise Exception(
                            "I thought all martin runs had a probe")
                else:
                    print("\tCouldn't recognize Probe performed val {} in file {}".format(
                        fieldVal, sesh.infoFileName))
            elif fieldName.lower() == "task ended at":
                sesh.btEndedAtWell = int(fieldVal)
            elif fieldName.lower() == "probe ended at":
                sesh.probeEndedAtWell = int(fieldVal)
            elif fieldName.lower() == "weight":
                sesh.weight = float(fieldVal)
            elif fieldName.lower() == "conditiongroup":
                cgs = fieldVal.strip().split("-")
                if not (len(cgs) == 2 and ("B" + cgs[0] == sesh.animalName or cgs[0] == sesh.animalName or f"{cgs[0]}_old" == sesh.animalName)):
                    print(fieldVal, cgs, sesh.animalName)
                    assert False
                sesh.conditionGroup = fieldVal
            elif fieldName.lower() == "probe home fill time":
                sesh.probeFillTime = int(fieldVal)
            elif fieldName.lower() == "trodes lights skip start":
                sesh.trodesLightsIgnoreSeconds = float(fieldVal)
            else:
                sesh.notes.append(line)

    if sesh.probePerformed is None and sesh.animalName == "B12":
        sesh.probePerformed = True

    if sesh.probePerformed and sesh.probeEndedAtWell is None:
        raise Exception(
            "Didn't mark the probe end well for sesh {}".format(sesh.name))

    if not (sesh.lastAwayWell == sesh.awayWells[-1] and sesh.endedOnHome) and sesh.btEndedAtWell is None:
        raise Exception(
            "Didn't find all the wells but task end well not marked for sesh {}".format(sesh.name))

    if sesh.prevInfoFileName is not None:
        with open(sesh.prevInfoFileName, 'r') as f:
            sesh.prevSessionInfoParsed = True
            lines = f.readlines()
            for line in lines:
                lineparts = line.split(":")
                if len(lineparts) != 2:
                    continue

                fieldName = lineparts[0]
                fieldVal = lineparts[1]

                if fieldName.lower() == "home":
                    sesh.prevSessionHome = int(fieldVal)
                elif fieldName.lower() == "aways":
                    sesh.prevSessionAways = [int(w)
                                             for w in fieldVal.strip().split(' ')]
                elif fieldName.lower() == "condition":
                    typeIn = fieldVal.lower()
                    if 'ripple' in typeIn or 'interruption' in typeIn:
                        # sesh.prevSessionIsRippleInterruption = True
                        sesh.prevSessionCondition = BTSession.CONDITION_INTERRUPTION
                    elif 'none' in typeIn:
                        # sesh.prevSessionIsNoInterruption = True
                        sesh.prevSessionCondition = BTSession.CONDITION_NO_STIM
                    elif 'delay' in typeIn:
                        # sesh.prevSessionIsDelayedInterruption = True
                        sesh.prevSessionCondition = BTSession.CONDITION_DELAY
                    else:
                        print("\tCouldn't recognize Condition {} in file {}".format(
                            typeIn, sesh.prevInfoFileName))
                elif fieldName.lower() == "thresh":
                    if "low" in fieldVal.lower():
                        sesh.prevSessionRippleDetectionThreshold = 2.5
                    elif "high" in fieldVal.lower():
                        sesh.prevSessionRippleDetectionThreshold = 4
                    elif "med" in fieldVal.lower():
                        sesh.prevSessionRippleDetectionThreshold = 3
                    elif "none" in fieldVal.lower():
                        sesh.prevSessionRippleDetectionThreshold = None
                    else:
                        sesh.prevSessionRippleDetectionThreshold = float(
                            fieldVal)
                elif fieldName.lower() == "last away":
                    if 'none' in fieldVal.lower():
                        sesh.prevSessionLastAwayWell = None
                    else:
                        sesh.prevSessionLastAwayWell = int(fieldVal)
                elif fieldName.lower() == "last well":
                    if 'h' in fieldVal.lower():
                        sesh.prevSessionEndedOnHome = True
                    elif 'a' in fieldVal.lower():
                        sesh.prevSessionEndedOnHome = False
                    elif 'none' in fieldVal.lower():
                        sesh.prevSessionEndedOnHome = None
                    else:
                        print("\tCouldn't recognize last well {} in file {}".format(
                            fieldVal, sesh.prevInfoFileName))
                elif fieldName.lower() == "iti stim on":
                    if 'Y' in fieldVal:
                        sesh.prevSessionItiStimOn = True
                    elif 'N' in fieldVal:
                        sesh.prevSessionItiStimOn = False
                    else:
                        print("\tCouldn't recognize ITI Stim condition {} in file {}".format(
                            fieldVal, sesh.prevInfoFileName))
                elif fieldName.lower() == "probe stim on":
                    if 'Y' in fieldVal:
                        sesh.prevSessionProbeStimOn = True
                    elif 'N' in fieldVal:
                        sesh.prevSessionProbeStimOn = False
                    else:
                        print("\tCouldn't recognize Probe Stim condition {} in file {}".format(
                            fieldVal, sesh.prevInfoFileName))
                else:
                    pass

    else:
        sesh.prevSessionInfoParsed = False

    if not sesh.importOptions.justExtractData:
        assert sesh.homeWell != 0

    if sesh.foundWells is None:
        sesh.foundWells = generateFoundWells(
            sesh.homeWell, sesh.awayWells, sesh.lastAwayWell, sesh.endedOnHome, sesh.foundFirstHome)


def parseDLCData(sesh: BTSession) -> Optional[Tuple[ArrayLike, ArrayLike, ArrayLike, Dict[str, Tuple[int, int]]]]:
    btPosFileName = sesh.loadInfo.DLC_dir + sesh.name + "_task.npz"
    probePosFileName = sesh.loadInfo.DLC_dir + sesh.name + "_probe.npz"
    if not os.path.exists(btPosFileName) or not os.path.exists(probePosFileName):
        print(f"\t{sesh.name} has no DLC file!")
        return None

    wellCoordsFileName = sesh.fileStartString + '.1.wellLocations_dlc.csv'
    if not os.path.exists(wellCoordsFileName):
        wellCoordsFileName = os.path.join(
            sesh.loadInfo.data_dir, 'well_locations_dlc.csv')
        print("\tSpecific well locations not found, falling back to file {}".format(
            wellCoordsFileName))
    wellCoordsMap = readWellCoordsFile(wellCoordsFileName)

    btPos = np.load(btPosFileName)
    sesh.btPosXs, sesh.btPosYs, sesh.btPos_ts = \
        processPosData(btPos["x1"], btPos["y1"],
                       btPos["timestamp"], wellCoordsMap, xLim=None, yLim=None, smooth=3)

    probePos = np.load(probePosFileName)
    sesh.probePosXs, sesh.probePosYs, sesh.probePos_ts = \
        processPosData(probePos["x1"], probePos["y1"],
                       probePos["timestamp"], wellCoordsMap, xLim=None, yLim=None, smooth=3)

    sesh.hasPositionData = True

    xs = np.hstack((sesh.btPosXs, sesh.probePosXs))
    ys = np.hstack((sesh.btPosYs, sesh.probePosYs))
    ts = np.hstack((sesh.btPos_ts, sesh.probePos_ts))

    return xs, ys, ts, wellCoordsMap


def cleanupPos(tpts: ArrayLike, xPos: ArrayLike, yPos: ArrayLike, xLim: Tuple[int, int] = None,
               yLim: Tuple[int, int] = None, excludeBoxes: Optional[List[Tuple[int, int, int, int]]] = None,
               maxJumpDistance: Optional[float] = None, makePlots: bool = False, minCleanTimeFrames: Optional[int] = None) -> None:
    # only in bounds points pls
    pointsInRange = np.ones_like(xPos).astype(bool)
    if xLim is not None:
        pointsInRange &= (xPos > xLim[0]) & (xPos < xLim[1])
    if yLim is not None:
        pointsInRange &= (yPos > yLim[0]) & (yPos < yLim[1])
    xPos[~ pointsInRange] = np.nan
    yPos[~ pointsInRange] = np.nan
    if makePlots:
        quickPosPlot(tpts, xPos, yPos, "in bounds only")

    # exclude boxes
    if excludeBoxes is not None:
        for x1, y1, x2, y2 in excludeBoxes:
            inBox = (xPos > x1) & (xPos < x2) & (yPos > y1) & (yPos < y2)
            xPos[inBox] = np.nan
            yPos[inBox] = np.nan
        if makePlots:
            quickPosPlot(tpts, xPos, yPos, "excluded boxes")

    # Remove large jumps in position (tracking errors)
    if maxJumpDistance is not None:
        jumpDistance = np.sqrt(np.square(np.diff(xPos, prepend=xPos[0])) +
                               np.square(np.diff(yPos, prepend=yPos[0])))
        cleanPoints = jumpDistance < maxJumpDistance
        xPos[~ cleanPoints] = np.nan
        yPos[~ cleanPoints] = np.nan
        if makePlots:
            quickPosPlot(tpts, xPos, yPos, "no jumps (single)")

    # Dilate the excluded parts but only inward toward the noise
    if minCleanTimeFrames is not None:
        nanidx = np.argwhere(np.isnan(xPos)).reshape(-1)
        nidiff = np.diff(nanidx)
        takeout = np.argwhere((nidiff > 1) & (nidiff < minCleanTimeFrames))
        for t in takeout:
            xPos[nanidx[t][0]:nanidx[t+1][0]] = np.nan
            yPos[nanidx[t][0]:nanidx[t+1][0]] = np.nan
        if makePlots:
            quickPosPlot(tpts, xPos, yPos, "no jumps (dilated)")


def interpNanPositions(tpts: ArrayLike, xPos: ArrayLike, yPos: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    nanpos = np.isnan(xPos)
    ynanpos = np.isnan(yPos)
    assert all(ynanpos == nanpos)
    notnanpos = np.logical_not(nanpos)
    xPos = np.interp(tpts, tpts[notnanpos], xPos[notnanpos])
    yPos = np.interp(tpts, tpts[notnanpos], yPos[notnanpos])
    return xPos, yPos


def getWellCameraCoordinates(well_num: int, well_coords_map: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    return well_coords_map[str(well_num)]


def correctFishEye(wellCoordsMap: Dict[str, Tuple[int, int]], xs: ArrayLike, ys: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    # returns position given by xs, ys in the context of sesh but corrected so
    # well locations would lay on a grid at x and y values 0.5, 1.5, ... 5.5

    # Here's the lattice we're mapping to, including imaginary wells that encircle the real wells
    xv, yv = np.meshgrid(np.linspace(-0.5, 6.5, 8), np.linspace(-0.5, 6.5, 8))
    xpts = xv.reshape((-1))
    ypts = yv.reshape((-1))

    def getWellCoords(xp, yp):
        if xp > 0 and xp < 6 and yp > 0 and yp < 6:
            wellName = int(2 + (xp - 0.5) + 8 * (yp - 0.5))
            return np.array(getWellCameraCoordinates(wellName, wellCoordsMap))
        if xp > 0 and xp < 6:
            # vertical extrapolation. p1 next well, p2 is further
            if yp > 6:
                p1 = getWellCoords(xp, yp-1)
                p2 = getWellCoords(xp, yp-2)
            else:
                p1 = getWellCoords(xp, yp+1)
                p2 = getWellCoords(xp, yp+2)
        else:
            if xp > 6:
                p1 = getWellCoords(xp-1, yp)
                p2 = getWellCoords(xp-2, yp)
            else:
                p1 = getWellCoords(xp+1, yp)
                p2 = getWellCoords(xp+2, yp)

        return p1 + (p1 - p2)

    cameraWellLocations = np.array([getWellCoords(x, y) for x, y in zip(xpts, ypts)])
    pathCoords = np.vstack((np.array(xs), np.array(ys))).T
    xres = griddata(cameraWellLocations, xpts, pathCoords, method="cubic")
    yres = griddata(cameraWellLocations, ypts, pathCoords, method="cubic")

    return xres, yres


def integrateCorrectionFiles(ts: ArrayLike, xs: ArrayLike, ys: ArrayLike, correctionDirectory: Optional[str],
                             wellCoordsMap: Dict[str, Tuple[int, int]], minCorrectionSecs=1.0) -> None:
    validTpts = ts[~np.isnan(xs)]
    notnandiff = np.diff(validTpts)
    MIN_CORRECTION_TS = minCorrectionSecs * TRODES_SAMPLING_RATE
    needsCorrection = np.argwhere((notnandiff >= MIN_CORRECTION_TS)).reshape(-1)
    correctionRegions = []
    correctedFlag = []
    # print("\tLarge gaps that are being interped:")
    for nc in needsCorrection:
        t1 = int(validTpts[nc])
        t2 = int(validTpts[nc+1])

        timeStr1 = timeStrForTrodesTimestamp(t1)
        timeStr2 = timeStrForTrodesTimestamp(t2)

        entry = (
            t1,
            t2,
            timeStr1,
            timeStr2,
            (t2-t1) / TRODES_SAMPLING_RATE
        )
        correctionRegions.append(entry)
        correctedFlag.append(False)

        # print("\t" + "\t".join([str(s) for s in entry]))

    if correctionDirectory is None:
        response = input("No corrections provided, interp all these gaps anyway (y/N)?")
        if response != "y":
            exit()
        return

    if not os.path.exists(correctionDirectory):
        os.makedirs(correctionDirectory)

    gl = os.path.join(correctionDirectory, '*.videoPositionTracking')
    cfiles = glob.glob(gl)
    # print(gl, cfiles)
    numCorrectionsIntegrated = 0
    for cf in cfiles:
        _, posdata = readRawPositionData(cf)
        ct = np.array(posdata["timestamp"]).astype(float)
        cx = np.array(posdata["x1"]).astype(float)
        cy = np.array(posdata["y1"]).astype(float)
        cx, cy = correctFishEye(wellCoordsMap, cx, cy)

        # Seems like it includes all the video timestamps, but just extends wherever tracking happened for
        # some reason
        cd = np.abs(np.diff(cx, prepend=cx[0])) + np.abs(np.diff(cy, prepend=cy[0]))
        nzcd = np.nonzero(cd)[0]
        ci1 = nzcd[0]
        ci2 = nzcd[-1]
        if False:
            quickPosPlot(ct, cx, cy, "incoming correction points")
            quickPosPlot(
                ct, cx, cy, "incoming correction points (just with tracking data)", irange=(ci1, ci2))
        ct = ct[ci1:ci2]
        cx = cx[ci1:ci2]
        cy = cy[ci1:ci2]
        # quickPosPlot(ct, cx, cy, "incoming correction points")

        posStart_ts = ct[0]
        posEnd_ts = ct[-1]
        # print(f"\tchecking {cf}\n\t\t{posStart_ts} - {posEnd_ts}")
        for entryi, entry in enumerate(correctionRegions):
            t1 = entry[0]
            t2 = entry[1]
            # print("\t\tagainst entry\t", t1, t2)
            if t1 > posStart_ts and t2 < posEnd_ts:
                correctedFlag[entryi] = True
                numCorrectionsIntegrated += 1
                # print("\tfound correction for", "\t".join([str(s) for s in entry]))
                # integrate this bit

                tpi1 = np.searchsorted(ts, t1)
                tpi2 = np.searchsorted(ts, t2)
                cpi1 = np.searchsorted(ct, t1)
                cpi2 = np.searchsorted(ct, t2)

                # MARGIN = 20
                # quickPosPlot(ct, cx, cy, "incoming correction points, full")
                # quickPosPlot(ts, xs, ys, "correction region original",
                #              irange=(max(0, tpi1 - MARGIN), min(len(ts)-1, tpi2 + MARGIN)))
                # quickPosPlot(ct, cx, cy, "incoming correction points", irange=(cpi1, cpi2))
                for t in ct:
                    assert t in ts
                for ci, pi in zip(range(cpi1, cpi2), range(tpi1, tpi2)):
                    if np.isnan(xs[pi]):
                        xs[pi] = cx[ci]
                        ys[pi] = cy[ci]
                # quickPosPlot(ts, xs, ys, "correction region integrated",
                # irange=(max(0, tpi1 - MARGIN), min(len(ts)-1, tpi2 + MARGIN)))

    print(f"\tCorrected {numCorrectionsIntegrated} regions with corrections files")
    # print("\tRemaining regions that are uncorrected:")
    lastEnd = None
    lastStart = None
    optimizedTimes = []
    SPLITGAP = 90
    COMBINEGAP = 30
    MAXREC = 60*5
    for entryi, entry in enumerate(correctionRegions):
        if not correctedFlag[entryi]:
            # print("\t" + "\t".join([str(s) for s in entry]))
            # oe format: (start time str, end time str, num included regions, length of biggest gap, gap start, gap end)

            if lastEnd is not None and (
                (entry[0] - lastEnd) / TRODES_SAMPLING_RATE < COMBINEGAP or
                ((entry[0] - lastEnd) / TRODES_SAMPLING_RATE < SPLITGAP and
                    (entry[1] - lastStart) / TRODES_SAMPLING_RATE < MAXREC)):
                # combine this with last entry
                currentOptimizedEntry = optimizedTimes[-1]
                gapLen = entry[0] - lastEnd
                if gapLen > currentOptimizedEntry[3]:
                    gapStart = currentOptimizedEntry[1]
                    gapEnd = entry[2]
                else:
                    gapLen = currentOptimizedEntry[3]
                    gapStart = currentOptimizedEntry[4]
                    gapEnd = currentOptimizedEntry[5]
                optimizedTimes[-1] = (currentOptimizedEntry[0], entry[3],
                                      currentOptimizedEntry[2]+1, gapLen, gapStart, gapEnd)
                lastEnd = entry[1]
            else:
                # new entry
                lastStart = entry[0]
                lastEnd = entry[1]
                optimizedTimes.append((entry[2], entry[3], 1, -1, None, None))

    # print("\toptimized corrections = ")
    # for oe in optimizedTimes:
    #     print(f"\t\t({oe[2]}) {oe[0]}\t{oe[1]}\t\t{oe[4]}\t{oe[5]}")

    correctionsFileName = os.path.join(correctionDirectory, "optimized.txt")
    # print(f"\tsaving optimized list to file {correctionsFileName}")
    # print("\tsaving optimized list to file")
    with open(correctionsFileName, 'w') as f:
        f.writelines([f"{oe[0]} - {oe[1]}\n" for oe in optimizedTimes])


def processPosData(x: ArrayLike, y: ArrayLike, t: ArrayLike, wellCoordsMap: Dict[str, Tuple[int, int]],
                   maxJumpDistance: Optional[float] = 0.25,
                   xLim: Tuple[int, int] = (100, 1050), yLim: Tuple[int, int] = (20, 900), smooth: Optional[float] = None,
                   excludeBoxes: Optional[List[Tuple[int, int, int, int]]] = None, correctionDirectory: Optional[str] = None,
                   minCleanTimeFrames: Optional[int] = 15) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    xPos = np.array(x, dtype=float)
    yPos = np.array(y, dtype=float)

    # Interpolate the position data into evenly sampled time points

    tpts = np.array(t).astype(float)
    xPos = np.array(x).astype(float)
    yPos = np.array(y).astype(float)

    # Remove out of bounds and excluded position points
    cleanupPos(tpts, xPos, yPos, xLim=xLim, yLim=yLim, excludeBoxes=excludeBoxes)

    # Run fisheye correction
    xPos, yPos = correctFishEye(wellCoordsMap, xPos, yPos)

    # Exclude large jumps
    cleanupPos(tpts, xPos, yPos, maxJumpDistance=maxJumpDistance,
               minCleanTimeFrames=minCleanTimeFrames)

    # Now check for large gaps that were removed and fill in with correction files
    integrateCorrectionFiles(tpts, xPos, yPos, correctionDirectory, wellCoordsMap)

    # Now interp all remaining unknown positions
    xPos, yPos = interpNanPositions(tpts, xPos, yPos)

    # quickPosPlot(tpts, xPos, yPos, "interp")

    if smooth is not None:
        xPos = gaussian_filter1d(xPos, smooth)
        yPos = gaussian_filter1d(yPos, smooth)

        # quickPosPlot(tpts, xPos, yPos, "smooth")

    return xPos, yPos, tpts


def getWellEntryAndExitTimes(nearestWells: ArrayLike, ts: ArrayLike,
                             quads: bool = False, includeNeighbors: bool = False) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    entryTimes = []
    exitTimes = []
    entryIdxs = []
    exitIdxs = []

    ts = np.append(np.array(ts), ts[-1] + (ts[-1] - ts[-2]))
    if quads:
        wellIdxs = [0, 1, 2, 3]
    else:
        wellIdxs = range(48)

    for wi in wellIdxs:
        # last data point should count as an exit, so appending a false
        # same for first point should count as entry, prepending
        if includeNeighbors:
            neighbors = list({wi, wi - 1, wi + 1, wi - 7, wi - 8, wi - 9,
                              wi + 7, wi + 8, wi + 9}.intersection(allWellNames))
            nearWell = np.concatenate(
                ([False], np.isin(nearestWells, neighbors), [False]))
        else:
            nearWell = np.concatenate(([False], nearestWells == wi, [False]))
        idx = np.argwhere(np.diff(np.array(nearWell, dtype=float)) == 1)
        # removing the -1 here so it's now exclusive exit time (exit and entrance will be the same, not 1 off anymore)
        idx2 = np.argwhere(np.diff(np.array(nearWell, dtype=float)) == -1)
        entryIdxs.append(idx.T[0])
        exitIdxs.append(idx2.T[0])
        entryTimes.append(ts[idx.T[0]])
        exitTimes.append(ts[idx2.T[0]])

    return entryIdxs, exitIdxs, entryTimes, exitTimes


def getNearestWell(xs: ArrayLike, ys: ArrayLike, well_idxs: ArrayLike = allWellNames, switchWellFactor: float = 0.8) -> ArrayLike:
    # switchWellFactor of 0.8 means transition from well a -> b requires rat dist to b to be 0.8 * dist to a
    xv, yv = np.meshgrid(np.linspace(0.5, 5.5, 6), np.linspace(0.5, 5.5, 6))
    xpts = xv.reshape((-1, 1))
    ypts = yv.reshape((-1, 1))
    keepWell = np.array([wn in well_idxs for wn in allWellNames])
    xpts = xpts[keepWell]
    ypts = ypts[keepWell]
    well_coords = np.hstack((xpts, ypts))

    tiled_x = np.tile(xs, (len(well_idxs), 1)).T  # each row is one time point
    tiled_y = np.tile(ys, (len(well_idxs), 1)).T

    tiled_wells_x = np.tile(well_coords[:, 0], (len(xs), 1))
    tiled_wells_y = np.tile(well_coords[:, 1], (len(ys), 1))

    delta_x = tiled_wells_x - tiled_x
    delta_y = tiled_wells_y - tiled_y
    delta = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))

    raw_nearest_wells = np.argmin(delta, axis=1)
    nearest_well = raw_nearest_wells
    curr_well = nearest_well[0]
    for i in range(np.shape(xs)[0]):
        if curr_well != nearest_well[i]:
            if delta[i, nearest_well[i]] < switchWellFactor * delta[i, curr_well]:
                curr_well = nearest_well[i]
            else:
                nearest_well[i] = curr_well

    # if TEST_NEAREST_WELL:
    #     print("delta_x", delta_x)
    #     print("delta_y", delta_y)
    #     print("delta", delta)
    #     print("raw_nearest_wells", raw_nearest_wells)
    #     print("nearest_well", nearest_well)

    return well_idxs[nearest_well]


def readClipData(data_filename: str) -> ArrayLike:
    time_clips = None
    try:
        with open(data_filename, 'r') as data_file:
            start_times = list()
            finish_times = list()
            csv_reader = csv.reader(data_file)
            n_time_clips = 0
            for data_row in csv_reader:
                if data_row:
                    n_time_clips += 1
                    start_times.append(int(data_row[1]))
                    finish_times.append(int(data_row[2]))
            time_clips = np.empty((n_time_clips, 2), dtype=np.uint32)
            time_clips[:, 0] = start_times[:]
            time_clips[:, 1] = finish_times[:]
    except Exception as err:
        print(err)
    return time_clips


def loadPositionData(sesh: BTSession) -> None:
    sesh.hasPositionData = False
    positionData = None

    # First try DeepLabCut
    if sesh.loadInfo.DLC_dir is not None:
        print(f"\tLoading from DLC, DLC_dir = {sesh.loadInfo.DLC_dir}")
        positionData = parseDLCData(sesh)
        if positionData is None:
            assert sesh.positionFromDeepLabCut is None or not sesh.positionFromDeepLabCut
            sesh.positionFromDeepLabCut = False
        else:
            assert sesh.positionFromDeepLabCut is None or sesh.positionFromDeepLabCut
            sesh.positionFromDeepLabCut = True
            xs, ys, ts, wellCoordsMap = positionData
            positionDataMetadata = {}
            sesh.hasPositionData = True
            raise Exception("Unimplemented new position standards for DLC (scale, fisheye, etc)")
    else:
        sesh.positionFromDeepLabCut = False

    if positionData is None:
        assert not sesh.positionFromDeepLabCut
        # print("\tUsing standard tracking, not DLC")
        trackingFile = sesh.fileStartString + '.1.videoPositionTracking'
        possibleOtherTrackingFile = sesh.fileStartString + '.1'
        if not os.path.exists(trackingFile) and os.path.exists(possibleOtherTrackingFile):
            print(
                f"\tFound a possible tracking file {possibleOtherTrackingFile}\n\tand no file {trackingFile}")
            if sesh.importOptions.skipConfirmTrackingFileExtension:
                print("\tAdding file extension automatically")
                renameFile = True
            else:
                response = input(
                    "Would you like to rename the .1 file to a .1.videoPositionTracking file (y/N)?")
                renameFile = response.strip() == "y"

            if renameFile:
                os.rename(possibleOtherTrackingFile, trackingFile)

        if os.path.exists(trackingFile):
            positionDataMetadata, positionData = readRawPositionData(
                trackingFile)

            if sesh.importOptions.forceAllTrackingAuto and \
                ("source" not in positionDataMetadata or
                 "trodescameraextractor" not in positionDataMetadata["source"]):
                print("\tForcing all tracking to be auto")
                positionData = None
                os.rename(sesh.fileStartString + '.1.videoPositionTracking', sesh.fileStartString +
                          '.1.videoPositionTracking.manualOutput')
            else:
                if "source" in positionDataMetadata and \
                        "trodescameraextractor" in positionDataMetadata["source"]:
                    print(
                        f"WARNING!!! {sesh.name} tracking was done with my script, not camera module")
                # print("\tFound existing tracking data")
                sesh.frameTimes = positionData['timestamp']
                correctionDirectory = os.path.join(os.path.dirname(
                    sesh.fileStartString), "trackingCorrections")
                # print(f"\tcorrectionDirectory is {correctionDirectory} ")

                wellCoordsFileName = sesh.fileStartString + '.1.wellLocations.csv'
                if not os.path.exists(wellCoordsFileName):
                    wellCoordsFileName = os.path.join(
                        sesh.loadInfo.data_dir, 'well_locations.csv')
                    # print("\tSpecific well locations not found, falling back to file {}".format(
                    #     wellCoordsFileName))
                wellCoordsMap = readWellCoordsFile(wellCoordsFileName)

                xs, ys, ts = processPosData(positionData["x1"], positionData["y1"], positionData["timestamp"],
                                            wellCoordsMap,
                                            xLim=(sesh.loadInfo.X_START,
                                                  sesh.loadInfo.X_FINISH),
                                            yLim=(sesh.loadInfo.Y_START,
                                                  sesh.loadInfo.Y_FINISH),
                                            excludeBoxes=sesh.loadInfo.excludeBoxes,
                                            correctionDirectory=correctionDirectory)
                sesh.hasPositionData = True
        else:
            positionData = None

        if positionData is None:
            raise Exception("Do everything by hand now!")
            print("\trunning trodes camera extractor!")
            processRawTrodesVideo(sesh.fileStartString + '.1.h264')
            positionDataMetadata, positionData = readRawPositionData(
                trackingFile)
            sesh.frameTimes = positionData['timestamp']
            xs, ys, ts = processPosData(positionData["x1"], positionData["y1"], positionData["timestamp"],
                                        wellCoordsMap,
                                        xLim=(sesh.loadInfo.X_START,
                                              sesh.loadInfo.X_FINISH),
                                        yLim=(sesh.loadInfo.Y_START,
                                              sesh.loadInfo.Y_FINISH),
                                        excludeBoxes=sesh.loadInfo.excludeBoxes)
            sesh.hasPositionData = True

    if not sesh.hasPositionData:
        print(f"!!!!!!!\tWARNING: {sesh.name} has no position data\t!!!!!!")
        return

    if sesh.importOptions.skipUSB:
        # print("!!!!!!\tWARNING: Skipping USB\t!!!!!!")
        pass
    else:
        if "lightonframe" in positionDataMetadata:
            sesh.trodesLightOnFrame = int(
                positionDataMetadata['lightonframe'])
            sesh.trodesLightOffFrame = int(
                positionDataMetadata['lightoffframe'])
            print("\tmetadata says trodes light frames {}, {} (/{})".format(sesh.trodesLightOffFrame,
                                                                            sesh.trodesLightOnFrame, len(ts)))
            sesh.trodesLightOnTime = sesh.frameTimes[sesh.trodesLightOnFrame]
            sesh.trodesLightOffTime = sesh.frameTimes[sesh.trodesLightOffFrame]
            print("\tmetadata file says trodes light timestamps {}, {} (/{})".format(
                sesh.trodesLightOffTime, sesh.trodesLightOnTime, len(ts)))
            # playFrames(sesh.fileStartString + '.1.h264', sesh.trodesLightOffFrame -
            #            20, sesh.trodesLightOffFrame + 20)
            # playFrames(sesh.fileStartString + '.1.h264', sesh.trodesLightOnFrame -
            #            20, sesh.trodesLightOnFrame + 20)
        else:
            # was a separate metadatafile made?
            positionMetadataFile = sesh.fileStartString + '.1.justLights'
            if os.path.exists(positionMetadataFile):
                lightInfo = np.fromfile(
                    positionMetadataFile, sep=",").astype(int)
                sesh.trodesLightOffTime = lightInfo[0]
                sesh.trodesLightOnTime = lightInfo[1]
                # print("\tjustlights file says trodes light timestamps {}, {} (/{})".format(
                #     sesh.trodesLightOffTime, sesh.trodesLightOnTime, len(ts)))
            else:
                print(f"\tdoing the lights with file {sesh.fileStartString + '.1.h264'}")
                sesh.trodesLightOffTime, sesh.trodesLightOnTime = getTrodesLightTimes(
                    sesh.fileStartString + '.1.h264', showVideo=False,
                    ignoreFirstSeconds=sesh.trodesLightsIgnoreSeconds)
                print("\ttrodesLightFunc says trodes light Time {}, {} (/{})".format(
                    sesh.trodesLightOffTime, sesh.trodesLightOnTime, len(ts)))

        # possibleRoots = ["/home/wcroughan/data/"]
        possibleRoots = [s for s in [
            getDrivePathByLabel(ss) for ss in [f"WDC{i}" for i in range(4, 13)]
        ] if s is not None]
        harold = getDrivePathByLabel("Harold")
        if harold is not None:
            possibleRoots.append(harold)

        possibleSubDirs = [
            os.path.join("Data", "labvideos", "labvideos", "trimmed", sesh.animalName),
            os.path.join("Backup", "labVideos", "labvideos", "trimmed", sesh.animalName),
            "videos",
            "lab_videos",
            sesh.animalName,
            os.path.join("videos", "B16-20"),
            os.path.join("videos", sesh.animalName),
            os.path.join("videos", "B16-20", "trimmed", sesh.animalName),
            os.path.join("videos", "labvideos", "trimmed", sesh.animalName)
        ]
        possibleDirectories = []
        for pr in possibleRoots:
            for psd in possibleSubDirs:
                p = os.path.join(pr, psd)
                if os.path.exists(p):
                    possibleDirectories.append(p)
        # print(f"{possibleDirectories = }")

        useSeshIdxDirectly = sesh.animalName == "B18" or sesh.dateStr > "20221100"
        sesh.usbVidFile = getUSBVideoFile(
            sesh.name, possibleDirectories, seshIdx=sesh.seshIdx, useSeshIdxDirectly=useSeshIdxDirectly)
        if sesh.usbVidFile is None:
            print("??????? usb vid file not found for session ", sesh.name)
            print(possibleDirectories, sesh.seshIdx, useSeshIdxDirectly)
            assert False
        else:
            # print("\tRunning USB light time analysis, file", sesh.usbVidFile)
            sesh.usbLightOffFrame, sesh.usbLightOnFrame = processUSBVideoData(
                sesh.usbVidFile, overwriteMode="loadOld", showVideo=False)
            # if sesh.usbLightOffFrame is None or sesh.usbLightOnFrame is None:
            #     raise Exception("exclude me pls")

        if not sesh.importOptions.justExtractData or sesh.importOptions.runInteractiveExtraction:
            clipsFileName = sesh.fileStartString + '.1.clips'
            if not os.path.exists(clipsFileName) and len(sesh.foundWells) > 0:
                print("\tclips file not found, gonna launch clips generator")

                allPosNearestWells = getNearestWell(xs, ys)
                _, _, allPosWellEntryTimes, allPosWellExitTimes = \
                    getWellEntryAndExitTimes(
                        allPosNearestWells, ts)

                ann = AnnotatorWindow(xs, ys, ts, sesh.trodesLightOffTime, sesh.trodesLightOnTime,
                                      sesh.usbVidFile, allPosWellEntryTimes, allPosWellExitTimes,
                                      sesh.foundWells, not sesh.probePerformed, sesh.fileStartString + '.1',
                                      wellCoordsMap)
                ann.resize(1600, 800)
                ann.show()
                importParentApp.exec()

    if sesh.separateProbeFile:
        probeFileStr = os.path.join(
            sesh.probeDir, os.path.basename(sesh.probeDir))
        btTimeClips = readClipData(
            sesh.fileStartString + '.1.clips')[0]
        probeTimeClips = readClipData(probeFileStr + '.1.clips')[0]
    else:
        timeClips = readClipData(sesh.fileStartString + '.1.clips')
        if timeClips is None:
            btTimeClips = None
            if sesh.probePerformed:
                probeTimeClips = None
        else:
            btTimeClips = timeClips[0]
            if sesh.probePerformed:
                probeTimeClips = timeClips[1]

    if btTimeClips is None:
        print("\t!!WARNING: Clips couldn't be loaded")
        if sesh.importOptions.justExtractData:
            return
        else:
            raise FileNotFoundError()

    # print("\tclips: {} (/{})".format(btTimeClips, len(ts)))
    btStart_posIdx = np.searchsorted(ts, btTimeClips[0])
    btEnd_posIdx = np.searchsorted(ts, btTimeClips[1])
    sesh.btPosXs = xs[btStart_posIdx:btEnd_posIdx]
    sesh.btPosYs = ys[btStart_posIdx:btEnd_posIdx]
    sesh.btPos_ts = ts[btStart_posIdx:btEnd_posIdx]

    if sesh.probePerformed:
        if sesh.separateProbeFile:
            assert not sesh.positionFromDeepLabCut
            _, positionData = readRawPositionData(
                probeFileStr + '.1.videoPositionTracking')
            xs, ys, ts = processPosData(positionData["x1"], positionData["y1"], positionData["timestamp"],
                                        wellCoordsMap,
                                        xLim=(sesh.loadInfo.X_START,
                                              sesh.loadInfo.X_FINISH),
                                        yLim=(sesh.loadInfo.Y_START,
                                              sesh.loadInfo.Y_FINISH),
                                        excludeBoxes=sesh.loadInfo.excludeBoxes)

        probeStart_posIdx = np.searchsorted(ts, probeTimeClips[0])
        probeEnd_posIdx = np.searchsorted(ts, probeTimeClips[1])
        sesh.probePosXs = xs[probeStart_posIdx:probeEnd_posIdx]
        sesh.probePosYs = ys[probeStart_posIdx:probeEnd_posIdx]
        sesh.probePos_ts = ts[probeStart_posIdx:probeEnd_posIdx]

    rewardClipsFile = sesh.fileStartString + '.1.rewardClips'
    if not os.path.exists(rewardClipsFile):
        rewardClipsFile = sesh.fileStartString + '.1.rewardclips'
    if not os.path.exists(rewardClipsFile):
        if sesh.numHomeFound > 0:
            print("!!!!!!\tWell find times not marked for session {}\t!!!!!!!".format(sesh.name))
            sesh.hasRewardFindTimes = False
        else:
            sesh.hasRewardFindTimes = True
    else:
        sesh.hasRewardFindTimes = True
        wellVisitTimes = readClipData(rewardClipsFile)
        sesh.homeRewardEnter_ts = wellVisitTimes[::2, 0]
        sesh.homeRewardExit_ts = wellVisitTimes[::2, 1]
        sesh.awayRewardEnter_ts = wellVisitTimes[1::2, 0]
        sesh.awayRewardExit_ts = wellVisitTimes[1::2, 1]

    if sesh.hasActivelinkLog:
        centerLoggedDetection_ts = (sesh.loggedDetections_ts[0] + sesh.loggedDetections_ts[-1]) / 2
        if sesh.probePerformed:
            lastTime = sesh.probePos_ts[-1]
        else:
            lastTime = sesh.btPos_ts[-1]
        if centerLoggedDetection_ts < sesh.btPos_ts[0] or centerLoggedDetection_ts > lastTime:
            raise Exception("Looks like we might have the wrong activelink log file.")


def loadLFPData(sesh: BTSession) -> Tuple[ArrayLike, Optional[ArrayLike]]:
    lfpData = []

    if len(sesh.rippleDetectionTetrodes) == 0:
        sesh.rippleDetectionTetrodes = [sesh.loadInfo.DEFAULT_RIP_DET_TET]

    # print(f"\tUsing detection tetrodes: {sesh.rippleDetectionTetrodes}")

    for i in range(len(sesh.rippleDetectionTetrodes)):
        lfpdir = sesh.fileStartString + ".LFP"
        if not os.path.exists(lfpdir):
            print("\t" + lfpdir, "\tdoesn't exists, gonna try and extract the LFP")
            # syscmd = "/home/wcroughan/SpikeGadgets/Trodes_1_8_1/exportLFP -rec " + sesh.fileStartString + ".rec"
            if os.path.exists("/home/wcroughan/Software/Trodes21/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            elif os.path.exists("/home/wcroughan/Software/Trodes21/linux/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/linux/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            # elif os.path.exists("/home/wcroughan/Software/Trodes_2-3-2_Ubuntu2004/exportLFP"):
            #     syscmd = "/home/wcroughan/Software/Trodes_2-3-2_Ubuntu2004/exportLFP -rec " + \
            #         sesh.fileStartString + ".rec"
            else:
                syscmd = "/home/wcroughan/Software/Trodes_2-2-3_Ubuntu1804/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            print("\t" + syscmd)
            os.system(syscmd)

        gl = os.path.join(lfpdir, sesh.name + ".LFP_nt" +
                          str(sesh.rippleDetectionTetrodes[i]) + "ch*.dat")
        lfpfilelist = glob.glob(gl)
        lfpfilename = lfpfilelist[0]
        sesh.btLfpFnames.append(lfpfilename)
        lfpData.append(MountainViewIO.loadLFP(
            data_file=sesh.btLfpFnames[-1]))

    if sesh.rippleBaselineTetrode is None:
        sesh.rippleBaselineTetrode = sesh.loadInfo.DEFAULT_RIP_BAS_TET

    # print(f"\tUsing baseline tetrode: {sesh.rippleBaselineTetrode}")

    # I think Martin didn't have this baseline tetrode? Need to check
    if sesh.rippleBaselineTetrode is not None:
        lfpdir = sesh.fileStartString + ".LFP"
        if not os.path.exists(lfpdir):
            print("\t" + lfpdir, "\tdoesn't exists, gonna try and extract the LFP")
            # syscmd = "/home/wcroughan/SpikeGadgets/Trodes_1_8_1/exportLFP -rec " + sesh.fileStartString + ".rec"
            if os.path.exists("/home/wcroughan/Software/Trodes21/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            # elif os.path.exists("/home/wcroughan/Software/Trodes_2-3-2_Ubuntu2004/exportLFP"):
            #     syscmd = "/home/wcroughan/Software/Trodes_2-3-2_Ubuntu2004/exportLFP -rec " + \
            #         sesh.fileStartString + ".rec"
            elif os.path.exists("/home/wcroughan/Software/Trodes21/linux/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/linux/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            else:
                syscmd = "/home/wcroughan/Software/Trodes_2-2-3_Ubuntu1804/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            print("\t" + syscmd)
            os.system(syscmd)

        gl = os.path.join(lfpdir, sesh.name + ".LFP_nt" +
                          str(sesh.rippleBaselineTetrode) + "ch*.dat")
        lfpfilelist = glob.glob(gl)
        lfpfilename = lfpfilelist[0]
        sesh.btLfpBaselineFname = lfpfilename
        baselineLfpData = MountainViewIO.loadLFP(
            data_file=sesh.btLfpBaselineFname)
    else:
        baselineLfpData = None

    return lfpData, baselineLfpData


def runLFPAnalyses(sesh: BTSession, lfpData: ArrayLike, baselineLfpData: Optional[ArrayLike], showPlot: bool = False) -> None:
    lfpV = lfpData[0][1]['voltage']
    lfp_ts = lfpData[0][0]['time'].astype(np.int64)
    # print("\tCondition - {}".format("SWR" if sesh.isRippleInterruption else "Ctrl"))

    peaks = signal.find_peaks(np.abs(np.diff(lfpV, prepend=lfpV[0])),
                              height=sesh.importOptions.DEFLECTION_THRESHOLD_HI, distance=sesh.importOptions.MIN_ARTIFACT_DISTANCE)
    lfpBumps_lfpIdx = peaks[0]
    sesh.lfpBumps_lfpIdx = lfpBumps_lfpIdx
    sesh.lfpBumps_ts = lfp_ts[lfpBumps_lfpIdx]
    sesh.btLFPBumps_posIdx = np.searchsorted(sesh.btPos_ts, sesh.lfpBumps_ts)
    sesh.btLFPBumps_posIdx = sesh.btLFPBumps_posIdx[(sesh.btLFPBumps_posIdx > 0) &
                                                    (sesh.btLFPBumps_posIdx < len(sesh.btPos_ts))]

    btLfpStart_lfpIdx = np.searchsorted(lfp_ts, sesh.btPos_ts[0])
    btLfpEnd_lfpIdx = np.searchsorted(lfp_ts, sesh.btPos_ts[-1])
    sesh.btLfpStart_lfpIdx = btLfpStart_lfpIdx
    sesh.btLfpEnd_lfpIdx = btLfpEnd_lfpIdx
    btLFPData = lfpV[btLfpStart_lfpIdx:btLfpEnd_lfpIdx]
    btLFPBumps_lfpIdx = lfpBumps_lfpIdx[(lfpBumps_lfpIdx > btLfpStart_lfpIdx) & (
        lfpBumps_lfpIdx < btLfpEnd_lfpIdx)]
    btLFPBumps_lfpIdx -= btLfpStart_lfpIdx
    sesh.btLFPBumps_lfpIdx = btLFPBumps_lfpIdx

    if sesh.hasActivelinkLog:
        sesh.loggedDetections_lfpIdx = np.searchsorted(lfp_ts, sesh.loggedDetections_ts)
        assert len(sesh.loggedDetections_lfpIdx) == len(sesh.loggedDetections_ts)
        sesh.btLoggedDetections_ts = sesh.loggedDetections_ts[(
            sesh.loggedDetections_ts > sesh.btPos_ts[0]) & (sesh.loggedDetections_ts < sesh.btPos_ts[-1])]
        sesh.btLoggedDetections_lfpIdx = sesh.loggedDetections_lfpIdx[(
            sesh.loggedDetections_lfpIdx > btLfpStart_lfpIdx) & (sesh.loggedDetections_lfpIdx < btLfpEnd_lfpIdx)]
        sesh.btLoggedDetections_lfpIdx -= btLfpStart_lfpIdx
        # print(f"{ len(sesh.btLoggedDetections_lfpIdx)  = }")
        # print(f"{ len(sesh.btLoggedDetections_ts) = }")
        assert len(sesh.btLoggedDetections_lfpIdx) == len(sesh.btLoggedDetections_ts)

    if sesh.probePerformed:
        itiMargin = sesh.importOptions.ITI_MARGIN
        sesh.itiLfpStart_ts = int(sesh.btPos_ts[-1] + itiMargin * TRODES_SAMPLING_RATE)
        sesh.itiLfpEnd_ts = int(sesh.probePos_ts[0] - itiMargin * TRODES_SAMPLING_RATE)
        itiLfpStart_lfpIdx = np.searchsorted(lfp_ts, sesh.itiLfpStart_ts)
        itiLfpEnd_lfpIdx = np.searchsorted(lfp_ts, sesh.itiLfpEnd_ts)
        sesh.itiLfpStart_lfpIdx = itiLfpStart_lfpIdx
        sesh.itiLfpEnd_lfpIdx = itiLfpEnd_lfpIdx
        itiLFPData = lfpV[itiLfpStart_lfpIdx:itiLfpEnd_lfpIdx]
        itiLFPBumps_lfpIdx = lfpBumps_lfpIdx[(lfpBumps_lfpIdx > itiLfpStart_lfpIdx) & (
            lfpBumps_lfpIdx < itiLfpEnd_lfpIdx)]
        itiLFPBumps_lfpIdx -= itiLfpStart_lfpIdx
        sesh.itiLFPBumps_lfpIdx = itiLFPBumps_lfpIdx

        if sesh.hasActivelinkLog:
            sesh.loggedDetections_lfpIdx = np.searchsorted(lfp_ts, sesh.loggedDetections_ts)
            sesh.itiLoggedDetections_ts = sesh.loggedDetections_ts[(
                sesh.loggedDetections_ts > sesh.itiLfpStart_ts) & (sesh.loggedDetections_ts < sesh.itiLfpEnd_ts)]
            sesh.itiLoggedDetections_lfpIdx = sesh.loggedDetections_lfpIdx[(
                sesh.loggedDetections_lfpIdx > itiLfpStart_lfpIdx) & (sesh.loggedDetections_lfpIdx < itiLfpEnd_lfpIdx)]
            sesh.itiLoggedDetections_lfpIdx -= sesh.itiLfpStart_lfpIdx
            assert len(sesh.itiLoggedDetections_lfpIdx) == len(sesh.itiLoggedDetections_ts)

        probeLfpStart_lfpIdx = np.searchsorted(lfp_ts, sesh.probePos_ts[0])
        probeLfpEnd_lfpIdx = np.searchsorted(lfp_ts, sesh.probePos_ts[-1])
        sesh.probeLfpStart_lfpIdx = probeLfpStart_lfpIdx
        sesh.probeLfpEnd_lfpIdx = probeLfpEnd_lfpIdx
        probeLFPData = lfpV[probeLfpStart_lfpIdx:probeLfpEnd_lfpIdx]
        probeLFPBumps_lfpIdx = lfpBumps_lfpIdx[(lfpBumps_lfpIdx > probeLfpStart_lfpIdx) & (
            lfpBumps_lfpIdx < probeLfpEnd_lfpIdx)]
        probeLFPBumps_lfpIdx -= probeLfpStart_lfpIdx
        sesh.probeLFPBumps_lfpIdx = probeLFPBumps_lfpIdx

        if sesh.hasActivelinkLog:
            assert probeLfpEnd_lfpIdx < len(lfp_ts)
            sesh.loggedDetections_lfpIdx = np.searchsorted(lfp_ts, sesh.loggedDetections_ts)
            sesh.probeLoggedDetections_ts = sesh.loggedDetections_ts[(
                sesh.loggedDetections_ts > sesh.probePos_ts[0]) & (sesh.loggedDetections_ts < sesh.probePos_ts[-1])]
            sesh.probeLoggedDetections_lfpIdx = sesh.loggedDetections_lfpIdx[(
                sesh.loggedDetections_lfpIdx > probeLfpStart_lfpIdx) & (sesh.loggedDetections_lfpIdx < probeLfpEnd_lfpIdx)]
            sesh.probeLoggedDetections_lfpIdx -= sesh.probeLfpStart_lfpIdx
            assert len(sesh.probeLoggedDetections_lfpIdx) == len(sesh.probeLoggedDetections_ts)

    peaks = signal.find_peaks(np.abs(
        np.diff(lfpV, prepend=lfpV[0])), height=sesh.importOptions.DEFLECTION_THRESHOLD_LO, distance=sesh.importOptions.MIN_ARTIFACT_DISTANCE)
    lfpNoise_lfpIdx = peaks[0]
    sesh.lfpNoise_ts = lfp_ts[lfpNoise_lfpIdx]
    sesh.lfpNoise_lfpIdx = lfpNoise_lfpIdx
    sesh.btLFPNoise_posIdx = np.searchsorted(sesh.btPos_ts, sesh.lfpNoise_ts)
    sesh.btLFPNoise_posIdx = sesh.btLFPNoise_posIdx[(sesh.btLFPNoise_posIdx > 0) &
                                                    (sesh.btLFPNoise_posIdx < len(sesh.btPos_ts))]
    btLFPNoise_lfpIdx = lfpNoise_lfpIdx[(lfpNoise_lfpIdx > btLfpStart_lfpIdx) & (
        lfpNoise_lfpIdx < btLfpEnd_lfpIdx)]
    btLFPNoise_lfpIdx -= btLfpStart_lfpIdx
    sesh.btLFPNoise_lfpIdx = btLFPNoise_lfpIdx

    if sesh.probePerformed:
        itiLFPNoise_lfpIdx = lfpNoise_lfpIdx[(lfpNoise_lfpIdx > itiLfpStart_lfpIdx) & (
            lfpNoise_lfpIdx < itiLfpEnd_lfpIdx)]
        itiLFPNoise_lfpIdx -= itiLfpStart_lfpIdx
        sesh.itiLFPNoise_lfpIdx = itiLFPNoise_lfpIdx

        probeLFPNoise_lfpIdx = lfpNoise_lfpIdx[(lfpNoise_lfpIdx > probeLfpStart_lfpIdx) & (
            lfpNoise_lfpIdx < probeLfpEnd_lfpIdx)]
        probeLFPNoise_lfpIdx -= probeLfpStart_lfpIdx
        sesh.probeLFPNoise_lfpIdx = probeLFPNoise_lfpIdx

    # Pre-bt stats:
    _, _, sesh.rpowmPreBt, sesh.rpowsPreBt = getRipplePower(
        lfpV[0:(btLfpStart_lfpIdx//2)], lfpDeflections=lfpNoise_lfpIdx)
    _, zpow, _, _ = getRipplePower(btLFPData, method="standard", meanPower=sesh.rpowmPreBt,
                                   stdPower=sesh.rpowsPreBt, lfpDeflections=sesh.btLFPNoise_lfpIdx)
    sesh.btRipsPreStats = detectRipples(
        zpow, lambda lfpIdx: lfp_ts[lfpIdx + sesh.btLfpStart_lfpIdx])

    if sesh.probePerformed:
        _, zpow, _, _ = getRipplePower(itiLFPData, method="standard", meanPower=sesh.rpowmPreBt,
                                       stdPower=sesh.rpowsPreBt, lfpDeflections=sesh.itiLFPNoise_lfpIdx)
        sesh.itiRipsPreStats = detectRipples(
            zpow, lambda lfpIdx: lfp_ts[lfpIdx + sesh.itiLfpStart_lfpIdx])
        _, zpow, _, _ = getRipplePower(probeLFPData, method="standard", meanPower=sesh.rpowmPreBt,
                                       stdPower=sesh.rpowsPreBt, lfpDeflections=sesh.probeLFPNoise_lfpIdx)
        sesh.probeRipsPreStats = detectRipples(
            zpow, lambda lfpIdx: lfp_ts[lfpIdx + sesh.probeLfpStart_lfpIdx])

        # probe stats
        _, _, sesh.rpowmProbe, sesh.rpowsProbe = getRipplePower(
            probeLFPData, lfpDeflections=probeLFPNoise_lfpIdx)
        _, zpow, _, _ = getRipplePower(btLFPData, method="standard", meanPower=sesh.rpowmProbe,
                                       stdPower=sesh.rpowsProbe, lfpDeflections=sesh.btLFPNoise_lfpIdx)
        sesh.btRipsProbeStats = detectRipples(
            zpow, lambda lfpIdx: lfp_ts[lfpIdx + sesh.btLfpStart_lfpIdx])
        _, zpow, _, _ = getRipplePower(itiLFPData, method="standard", meanPower=sesh.rpowmProbe,
                                       stdPower=sesh.rpowsProbe, lfpDeflections=sesh.itiLFPNoise_lfpIdx)
        sesh.itiRipsProbeStats = detectRipples(
            zpow, lambda lfpIdx: lfp_ts[lfpIdx + sesh.itiLfpStart_lfpIdx])
        _, zpow, _, _ = getRipplePower(probeLFPData, method="standard", meanPower=sesh.rpowmProbe,
                                       stdPower=sesh.rpowsProbe, lfpDeflections=sesh.probeLFPNoise_lfpIdx)
        sesh.probeRipsProbeStats = detectRipples(
            zpow, lambda lfpIdx: lfp_ts[lfpIdx + sesh.probeLfpStart_lfpIdx])

    if sesh.hasActivelinkLog:
        if sesh.btLfpBaselineFname is None:
            raise Exception("Session has activelink log but not a basline lfp")
        baselineLfpV = baselineLfpData[1]['voltage']
        _, zpow, _, _ = getRipplePower(btLFPData, method="activelink", meanPower=sesh.rpowmLog,
                                       stdPower=sesh.rpowsLog, lfpDeflections=sesh.btLFPNoise_lfpIdx,
                                       baselineLfpData=baselineLfpV[btLfpStart_lfpIdx:btLfpEnd_lfpIdx])
        sesh.btRipsLogStats = detectRipples(
            zpow, lambda lfpIdx: lfp_ts[lfpIdx + sesh.btLfpStart_lfpIdx])
        _, zpow, _, _ = getRipplePower(itiLFPData, method="activelink", meanPower=sesh.rpowmLog,
                                       stdPower=sesh.rpowsLog, lfpDeflections=sesh.itiLFPNoise_lfpIdx,
                                       baselineLfpData=baselineLfpV[itiLfpStart_lfpIdx:itiLfpEnd_lfpIdx])
        sesh.itiRipsLogStats = detectRipples(
            zpow, lambda lfpIdx: lfp_ts[lfpIdx + sesh.itiLfpStart_lfpIdx])
        _, zpow, _, _ = getRipplePower(probeLFPData, method="activelink", meanPower=sesh.rpowmLog,
                                       stdPower=sesh.rpowsLog, lfpDeflections=sesh.probeLFPNoise_lfpIdx,
                                       baselineLfpData=baselineLfpV[probeLfpStart_lfpIdx:probeLfpEnd_lfpIdx])
        sesh.probeRipsLogStats = detectRipples(
            zpow, lambda lfpIdx: lfp_ts[lfpIdx + sesh.probeLfpStart_lfpIdx])


def runSanityChecks(sesh: BTSession, lfpData: ArrayLike, baselineLfpData: Optional[ArrayLike], showPlots: bool = False, overrideNotes: bool = True) -> None:
    # print(f"Running sanity checks for {sesh.name}")
    if lfpData is None:
        print("\tNo LFP data to look at")
    else:
        # lfpV = lfpData[0][1]['voltage']
        lfp_ts = lfpData[0][0]['time']

        # lfpDeflections = signal.find_peaks(np.abs(np.diff(
        #     lfpV, prepend=lfpV[0])), height=C["DEFLECTION_THRESHOLD_HI"], distance=C["MIN_ARTIFACT_DISTANCE"])
        # interruptions_lfpIdx = lfpDeflections[0]
        # interruption_ts = lfp_ts[interruptions_lfpIdx]
        # btInterruption_posIdx = np.searchsorted(sesh.btPos_ts, interruption_ts)
        # btInterruption_posIdx = sesh.btInterruption_posIdx[btInterruption_posIdx < len(
        #     sesh.btPos_ts)]

        numInterruptions = len(sesh.btLFPBumps_lfpIdx)
        print("\t{} interruptions detected".format(numInterruptions))
        if numInterruptions < 50:
            if sesh.isRippleInterruption and numInterruptions > 0:
                print("WARNING: FEWER THAN 50 STIMS DETECTED ON AN INTERRUPTION SESSION")
            elif numInterruptions == 0:
                print("WARNING: IGNORING BEHAVIOR NOTES FILE BECAUSE SEEING "
                      "0 INTERRUPTIONS, CALLING THIS A CONTROL SESSION")
                if overrideNotes:
                    # sesh.isRippleInterruption = False
                    sesh.condition = BTSession.CONDITION_NO_STIM
            else:
                print(
                    "WARNING: very few interruptions. This was a delay control but is basically a no-stim control")
        elif numInterruptions < 100:
            print("\t50-100 interruptions: not overriding label")

        dt = np.diff(lfp_ts)
        gapThresh = 2.0 * float(TRODES_SAMPLING_RATE / LFP_SAMPLING_RATE)
        isBigGap = dt > gapThresh

        if not any(isBigGap):
            print("\tNo gaps in LFP!")
        else:
            totalTime = (lfp_ts[-1] - lfp_ts[0]) / TRODES_SAMPLING_RATE
            totalGapTime = np.sum(dt[isBigGap])
            print(f"\t{totalGapTime}/{totalTime} ({int(100*totalGapTime/totalTime)}%) of lfp signal missing")

            maxGapIdx = np.argmax(dt)
            maxGapLen = dt[maxGapIdx] / TRODES_SAMPLING_RATE
            maxGapT1 = (lfp_ts[maxGapIdx] - lfp_ts[0]) / TRODES_SAMPLING_RATE
            maxGapT2 = (lfp_ts[maxGapIdx + 1] - lfp_ts[0]) / TRODES_SAMPLING_RATE
            print(f"\tBiggest gap: {maxGapLen}s long ({maxGapT1} - {maxGapT2})")

    # dummySesh = BTSession()
    # dummyDict = dummySesh.__dict__
    # for k in sesh.__dict__:
    #     v = sesh.__dict__[k]
    #     if k not in dummyDict:
    #         print(f"Found attribute {k} with value type {type(v)}")
    #     # elif not isinstance(v, type(dummyDict[k])) and dummyDict[k] is not None:
    #     #     print(f"Attribute {k} changed type:\t{type(dummyDict[k])}\t{type(v)}!")
    #     elif isinstance(v, (list, np.ndarray)):
    #         # print(k, v)
    #         same = np.array(v) == np.array(dummyDict[k])
    #         if isinstance(same, bool) and same:
    #             print(f"Attribute {k} was never changed")
    #     elif v == dummyDict[k]:
    #         print(f"Attribute {k} was never changed")


@TimeThisFunction
def posCalcVelocity(sesh: BTSession) -> None:
    btVel = np.sqrt(np.power(np.diff(sesh.btPosXs), 2) +
                    np.power(np.diff(sesh.btPosYs), 2))

    oldSettings = np.seterr(invalid="ignore")
    sesh.btVelCmPerSRaw = np.divide(btVel, np.diff(sesh.btPos_ts) /
                                    TRODES_SAMPLING_RATE) * CM_PER_FT
    np.seterr(**oldSettings)
    POS_FRAME_RATE = stats.mode(np.diff(sesh.btPos_ts), keepdims=True)[
        0] / float(TRODES_SAMPLING_RATE)
    BOUT_VEL_SM_SIGMA = sesh.importOptions.MOVE_THRESH_SM_SIGMA_SECS / POS_FRAME_RATE
    # print(f"{BOUT_VEL_SM_SIGMA =}")
    sesh.btVelCmPerS = gaussian_filter1d(sesh.btVelCmPerSRaw, BOUT_VEL_SM_SIGMA)
    btIsMv = sesh.btVelCmPerS > sesh.importOptions.VEL_THRESH
    if len(btIsMv) > 0:
        btIsMv = np.append(btIsMv, np.array(btIsMv[-1]))
    sesh.btIsMv = btIsMv

    if sesh.probePerformed:
        probeVel = np.sqrt(np.power(np.diff(sesh.probePosXs), 2) +
                           np.power(np.diff(sesh.probePosYs), 2))
        oldSettings = np.seterr(invalid="ignore")
        sesh.probeVelCmPerSRaw = np.divide(probeVel, np.diff(sesh.probePos_ts) /
                                           TRODES_SAMPLING_RATE) * CM_PER_FT
        np.seterr(**oldSettings)
        sesh.probeVelCmPerS = gaussian_filter1d(sesh.probeVelCmPerSRaw, BOUT_VEL_SM_SIGMA)
        probeIsMv = sesh.probeVelCmPerS > sesh.importOptions.VEL_THRESH
        if len(probeIsMv) > 0:
            probeIsMv = np.append(probeIsMv, np.array(probeIsMv[-1]))
        sesh.probeIsMv = probeIsMv


@TimeThisFunction
def posCalcEntryExitTimes(sesh: BTSession) -> None:
    # ===================================
    # Well and quadrant entry and exit times
    # ===================================
    sesh.btNearestWells = getNearestWell(sesh.btPosXs, sesh.btPosYs)

    sesh.btQuadrants = np.array(
        [quadrantOfWell(wi) for wi in sesh.btNearestWells])

    sesh.btWellEntryTimes_posIdx, sesh.btWellExitTimes_posIdx, \
        sesh.btWellEntryTimes_ts, sesh.btWellExitTimes_ts = \
        getWellEntryAndExitTimes(sesh.btNearestWells, sesh.btPos_ts)

    # ninc stands for neighbors included
    sesh.btWellEntryTimesNinc_posIdx, sesh.btWellExitTimesNinc_posIdx, \
        sesh.btWellEntryTimesNinc_ts, sesh.btWellExitTimesNinc_ts = \
        getWellEntryAndExitTimes(sesh.btNearestWells, sesh.btPos_ts, includeNeighbors=True)

    sesh.btQuadrantEntryTimes_posIdx, sesh.btQuadrantExitTimes_posIdx, \
        sesh.btQuadrantEntryTimes_ts, sesh.btQuadrantExitTimes_ts = \
        getWellEntryAndExitTimes(
            sesh.btQuadrants, sesh.btPos_ts, quads=True)

    for i in allWellNames:
        assert len(sesh.btWellEntryTimes_ts[i]) == len(sesh.btWellExitTimes_ts[i])

    # same for during probe
    if sesh.probePerformed:
        sesh.probeNearestWells = getNearestWell(sesh.probePosXs, sesh.probePosYs)

        sesh.probeWellEntryTimes_posIdx, sesh.probeWellExitTimes_posIdx, \
            sesh.probeWellEntryTimes_ts, sesh.probeWellExitTimes_ts = getWellEntryAndExitTimes(
                sesh.probeNearestWells, sesh.probePos_ts)
        sesh.probeWellEntryTimesNinc_posIdx, sesh.probeWellExitTimesNinc_posIdx, \
            sesh.probeWellEntryTimesNinc_ts, sesh.probeWellExitTimesNinc_ts = getWellEntryAndExitTimes(
                sesh.probeNearestWells, sesh.probePos_ts, includeNeighbors=True)

        sesh.probeQuadrants = np.array(
            [quadrantOfWell(wi) for wi in sesh.probeNearestWells])

        sesh.probeQuadrantEntryTimes_posIdx, sesh.probeQuadrantExitTimes_posIdx, \
            sesh.probeQuadrantEntryTimes_ts, sesh.probeQuadrantExitTimes_ts = getWellEntryAndExitTimes(
                sesh.probeQuadrants, sesh.probePos_ts, quads=True)

        for i in allWellNames:
            assert len(sesh.probeWellEntryTimes_ts[i]) == len(sesh.probeWellExitTimes_ts[i])

    homeMiddle_ts = (sesh.homeRewardEnter_ts + sesh.homeRewardExit_ts) / 2
    maxCorrection = 0
    numCorrections = 0
    numCorrectionsOutofrange = 0
    for ti, ts in enumerate(homeMiddle_ts):
        # print(ti, ts)
        # print(sesh.btWellEntryTimes_ts[sesh.homeWell], sesh.btWellExitTimes_ts[sesh.homeWell])
        nz = np.nonzero((sesh.btWellEntryTimes_ts[sesh.homeWell] < ts) & (
            sesh.btWellExitTimes_ts[sesh.homeWell] > ts))
        if len(nz) != 1:
            print(nz)
            print(sesh.btWellEntryTimes_ts[sesh.homeWell], sesh.btWellExitTimes_ts[sesh.homeWell])
        assert len(nz) == 1
        nz = nz[0]
        # print(f"{nz = }")
        if len(nz) != 1:
            # print(f"{ nz=  }")
            print(sesh.btWellEntryTimes_ts[sesh.homeWell], sesh.btWellExitTimes_ts[sesh.homeWell])
            print(ts)
            if len(sesh.btWellEntryTimes_ts[sesh.homeWell]) == 0:
                t1 = np.searchsorted(sesh.btPos_ts, sesh.homeRewardEnter_ts[ti]) - 20
                t2 = np.searchsorted(sesh.btPos_ts, sesh.homeRewardExit_ts[ti]) + 20
                # t3 = np.searchsorted(
                #     sesh.btPos_ts, sesh.btWellEntryTimes_ts[sesh.homeWell][nearestEntryTimeIdx]) - 20
                # t4 = np.searchsorted(
                #     sesh.btPos_ts, sesh.btWellExitTimes_ts[sesh.homeWell][nearestEntryTimeIdx]) + 20
                homex, homey = getWellPosCoordinates(sesh.homeWell)
                # plt.plot(sesh.btPosXs, sesh.btPosYs)
                # plt.plot(sesh.btPosXs[t1:t2], sesh.btPosYs[t1:t2])
                # # plt.plot(sesh.btPosXs[t3:t4], sesh.btPosYs[t3:t4])
                # plt.scatter(homex, homey, color='red')
                # plt.show()
                raise Exception("No home well visits found")
            # print(sesh.homeRewardEnter_ts[ti], sesh.homeRewardExit_ts[ti])
            nearestEntryTimeIdx = np.argmin(abs(sesh.btWellEntryTimes_ts[sesh.homeWell] - ts))
            print("!!!!!! WARNING: CORRECTING HOME REWARD TIME OUTSIDE OF VISIT !!!!!!!")
            nz = [nearestEntryTimeIdx]
        assert len(nz) == 1
        encompassingVisitIdx = nz[0]
        if sesh.homeRewardEnter_ts[ti] < \
                sesh.btWellEntryTimes_ts[sesh.homeWell][encompassingVisitIdx]:
            # print(f"Note: Fixing home well entry time {ti} for session {sesh.name}")
            correction = abs(sesh.homeRewardEnter_ts[ti] -
                             sesh.btWellEntryTimes_ts[sesh.homeWell][encompassingVisitIdx]) / TRODES_SAMPLING_RATE
            if correction > maxCorrection:
                maxCorrection = correction
            numCorrections += 1

            sesh.homeRewardEnter_ts[ti] = \
                sesh.btWellEntryTimes_ts[sesh.homeWell][encompassingVisitIdx]
        if sesh.homeRewardExit_ts[ti] > sesh.btWellExitTimes_ts[sesh.homeWell][encompassingVisitIdx]:
            # print(f"Note: Fixing home well Exit time {ti} for session {sesh.name}")
            correction = abs(sesh.homeRewardExit_ts[ti] -
                             sesh.btWellExitTimes_ts[sesh.homeWell][encompassingVisitIdx]) / TRODES_SAMPLING_RATE
            if correction > maxCorrection:
                maxCorrection = correction
            numCorrections += 1
            sesh.homeRewardExit_ts[ti] = sesh.btWellExitTimes_ts[sesh.homeWell][encompassingVisitIdx]

    awayMiddle_ts = (sesh.awayRewardEnter_ts + sesh.awayRewardExit_ts) / 2
    for ti, (ts, aw) in enumerate(zip(awayMiddle_ts, sesh.visitedAwayWells)):
        nz = np.nonzero((sesh.btWellEntryTimes_ts[aw] < ts) & (sesh.btWellExitTimes_ts[aw] > ts))
        assert len(nz) == 1
        nz = nz[0]
        if len(nz) == 0:
            # print(f"{aw = }")
            # print(f"{ti = }")
            # print(f"{ts = }")
            closest = np.argmin(np.abs(sesh.btWellEntryTimes_ts[aw] - ts))
            t1 = sesh.btWellEntryTimes_ts[aw][closest]
            t2 = sesh.btWellExitTimes_ts[aw][closest]
            # print(f"{t1 = }")
            # print(f"{t2 = }")
            if min(np.abs(ts - t1), np.abs(ts - t2)) < TRODES_SAMPLING_RATE:
                print("Found a close enough one with a second, using that")
                encompassingVisitIdx = closest
                numCorrectionsOutofrange += 1
            else:
                awayx, awayy = getWellPosCoordinates(aw)
                idx1 = np.searchsorted(sesh.btPos_ts, t1)
                idx2 = np.searchsorted(sesh.btPos_ts, t2)
                # plt.plot(sesh.btPosXs, sesh.btPosYs)
                # plt.plot(sesh.btPosXs[idx1:idx2], sesh.btPosYs[idx1:idx2])
                # plt.scatter(awayx, awayy, color='red')
                # plt.show()
                raise Exception("COuldn't match up with hand-marked visit with the detected visits")
        else:
            assert len(nz) == 1
            encompassingVisitIdx = nz[0]
        if sesh.awayRewardEnter_ts[ti] < sesh.btWellEntryTimes_ts[aw][encompassingVisitIdx]:
            # print(f"Note: Fixing away well entry time {ti} for session {sesh.name}")
            correction = abs(
                sesh.awayRewardEnter_ts[ti] - sesh.btWellEntryTimes_ts[aw][encompassingVisitIdx]) / TRODES_SAMPLING_RATE
            if correction > maxCorrection:
                maxCorrection = correction
            numCorrections += 1
            sesh.awayRewardEnter_ts[ti] = sesh.btWellEntryTimes_ts[aw][encompassingVisitIdx]
        if sesh.awayRewardExit_ts[ti] > sesh.btWellExitTimes_ts[aw][encompassingVisitIdx]:
            # print(f"Note: Fixing away well Exit time {ti} for session {sesh.name}")
            correction = abs(
                sesh.awayRewardExit_ts[ti] - sesh.btWellExitTimes_ts[aw][encompassingVisitIdx]) / TRODES_SAMPLING_RATE
            if correction > maxCorrection:
                maxCorrection = correction
            numCorrections += 1
            sesh.awayRewardExit_ts[ti] = sesh.btWellExitTimes_ts[aw][encompassingVisitIdx]

    print(f"\tFixed {numCorrections} well find times, max correction was {maxCorrection} seconds")

    sesh.homeRewardEnter_posIdx = np.searchsorted(
        sesh.btPos_ts, sesh.homeRewardEnter_ts)
    sesh.homeRewardExit_posIdx = np.searchsorted(
        sesh.btPos_ts, sesh.homeRewardExit_ts)
    sesh.awayRewardEnter_posIdx = np.searchsorted(
        sesh.btPos_ts, sesh.awayRewardEnter_ts)
    sesh.awayRewardExit_posIdx = np.searchsorted(
        sesh.btPos_ts, sesh.awayRewardExit_ts)


def getCurvature(x: ArrayLike, y: ArrayLike, H: float) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    # Knot-path-curvature as in https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000638
    H2 = H * H
    i1 = np.empty_like(x)
    i2 = np.empty_like(x)
    dxf = np.empty_like(x)
    dyf = np.empty_like(x)
    dxb = np.empty_like(x)
    dyb = np.empty_like(x)
    magf = np.empty_like(x)
    magb = np.empty_like(x)

    dx = np.diff(x)
    dy = np.diff(y)

    for pi in range(len(x)):
        # print(pi, end="\t")
        # forward
        ii = pi
        dxfi = 0.0
        dyfi = 0.0
        while ii < len(dx):
            dxfi += dx[ii]
            dyfi += dy[ii]
            magfi = dxfi * dxfi + dyfi * dyfi
            if magfi >= H2:
                # print(ii - pi)
                break
            ii += 1

        if ii == len(dx):
            i1[pi] = np.nan
            i2[pi] = np.nan
            dxf[pi] = np.nan
            dyf[pi] = np.nan
            dxb[pi] = np.nan
            dyb[pi] = np.nan
            magf[pi] = np.nan
            magb[pi] = np.nan
            continue
        i2[pi] = ii

        ii = pi - 1
        dxbi = 0.0
        dybi = 0.0
        while ii >= 0:
            dxbi += dx[ii]
            dybi += dy[ii]
            magbi = dxbi * dxbi + dybi * dybi
            if magbi >= H2:
                break
            ii -= 1

        if ii == -1:
            i1[pi] = np.nan
            i2[pi] = np.nan
            dxf[pi] = np.nan
            dyf[pi] = np.nan
            dxb[pi] = np.nan
            dyb[pi] = np.nan
            magf[pi] = np.nan
            magb[pi] = np.nan
            continue
        i1[pi] = ii

        dxf[pi] = dxfi
        dyf[pi] = dyfi
        dxb[pi] = dxbi
        dyb[pi] = dybi
        magf[pi] = magfi
        magb[pi] = magbi

    oldSettings = np.seterr(divide="ignore", invalid="ignore")

    uxf = dxf / np.sqrt(magf)
    uyf = dyf / np.sqrt(magf)
    uxb = dxb / np.sqrt(magb)
    uyb = dyb / np.sqrt(magb)
    dotprod = uxf * uxb + uyf * uyb
    curvature = np.arccos(dotprod)

    np.seterr(**oldSettings)

    return curvature, i1, i2, dxf, dyf, dxb, dyb


@TimeThisFunction
def posCalcCurvature(sesh: BTSession) -> None:
    KNOT_H_POS = sesh.importOptions.KNOT_H_CM / CM_PER_FT

    sesh.btCurvature, sesh.btCurvatureI1, sesh.btCurvatureI2, sesh.btCurvatureDxf, sesh.btCurvatureDyf, \
        sesh.btCurvatureDxb, sesh.btCurvatureDyb = getCurvature(
            sesh.btPosXs, sesh.btPosYs, KNOT_H_POS)

    if sesh.probePerformed:
        sesh.probeCurvature, sesh.probeCurvatureI1, sesh.probeCurvatureI2, sesh.probeCurvatureDxf, \
            sesh.probeCurvatureDyf, sesh.probeCurvatureDxb, sesh.probeCurvatureDyb = getCurvature(
                sesh.probePosXs, sesh.probePosYs, KNOT_H_POS)


def getExplorationCategories(ts, vel, nearestWells, importOptions, forcePauseIntervals=None) -> \
        Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    POS_FRAME_RATE = stats.mode(np.diff(ts), keepdims=True)[0] / float(TRODES_SAMPLING_RATE)
    BOUT_VEL_SM_SIGMA = importOptions.BOUT_VEL_SM_SIGMA_SECS / POS_FRAME_RATE
    MIN_PAUSE_TIME_FRAMES = int(importOptions.MIN_PAUSE_TIME_BETWEEN_BOUTS_SECS / POS_FRAME_RATE)
    MIN_EXPLORE_TIME_FRAMES = int(importOptions.MIN_EXPLORE_TIME_SECS / POS_FRAME_RATE)

    smoothVel = gaussian_filter1d(vel, BOUT_VEL_SM_SIGMA)

    isExploreLocal = smoothVel > importOptions.PAUSE_MAX_SPEED_CM_S
    dilationFilter = np.ones((MIN_PAUSE_TIME_FRAMES), dtype=int)
    inPauseBout = ~ (signal.convolve(isExploreLocal.astype(int),
                     dilationFilter, mode='same').astype(bool))
    isInPause = signal.convolve(inPauseBout.astype(int), dilationFilter, mode='same').astype(bool)
    isInExplore = np.logical_not(isInPause)

    if forcePauseIntervals is not None:
        for i1, i2 in forcePauseIntervals:
            pidx1 = np.searchsorted(ts[0:-1], i1)
            pidx2 = np.searchsorted(ts[0:-1], i2)
            isInPause[pidx1:pidx2] = True
            isInExplore[pidx1:pidx2] = False

    assert np.sum(np.isnan(isInPause)) == 0
    assert np.sum(np.isnan(isInExplore)) == 0
    assert np.all(np.logical_or(isInPause, isInExplore))
    assert not np.any(np.logical_and(isInPause, isInExplore))

    startExplores = np.where(np.diff(isInExplore.astype(int)) == 1)[0] + 1
    if isInExplore[0]:
        startExplores = np.insert(startExplores, 0, 0)

    stopExplores = np.where(np.diff(isInExplore.astype(int)) == -1)[0] + 1
    if isInExplore[-1]:
        stopExplores = np.append(stopExplores, len(isInExplore))

    boutLenFrames = stopExplores - startExplores
    longEnough = boutLenFrames >= MIN_EXPLORE_TIME_FRAMES

    boutNumWellsVisited = np.zeros((len(startExplores)))
    for i, (bst, ben) in enumerate(zip(startExplores, stopExplores)):
        boutNumWellsVisited[i] = len(
            getListOfVisitedWells(nearestWells[bst:ben], True))
    enoughWells = boutNumWellsVisited >= importOptions.MIN_EXPLORE_NUM_WELLS

    keepBout = np.logical_and(longEnough, enoughWells)
    exploreBoutStarts = startExplores[keepBout]
    exploreBoutEnds = stopExplores[keepBout]
    ts = np.array(ts)
    exploreBoutLensSecs = (ts[exploreBoutEnds] - ts[exploreBoutStarts]) / TRODES_SAMPLING_RATE

    # add a category at each behavior time point for easy reference later:
    boutCategory = np.zeros_like(ts)
    lastStop = 0
    for bst, ben in zip(exploreBoutStarts, exploreBoutEnds):
        boutCategory[lastStop:bst] = 1
        lastStop = ben
    boutCategory[lastStop:] = 1
    if forcePauseIntervals is not None:
        for i1, i2 in forcePauseIntervals:
            pidx1 = np.searchsorted(ts[0:-1], i1)
            pidx2 = np.searchsorted(ts[0:-1], i2)
            boutCategory[pidx1:pidx2] = 2

    boutLabel = np.zeros_like(ts)
    for bi, (bst, ben) in enumerate(zip(exploreBoutStarts, exploreBoutEnds)):
        boutLabel[bst:ben] = bi + 1

    return smoothVel, exploreBoutStarts, exploreBoutEnds, \
        exploreBoutLensSecs, boutCategory, boutLabel


@TimeThisFunction
def posCalcExplorationBouts(sesh: BTSession) -> None:
    sesh.btSmoothVel, sesh.btExploreBoutStart_posIdx, sesh.btExploreBoutEnd_posIdx, \
        sesh.btExploreBoutLensSecs, sesh.btBoutCategory, sesh.btBoutLabel = \
        getExplorationCategories(sesh.btPos_ts, sesh.btVelCmPerSRaw, sesh.btNearestWells, sesh.importOptions,
                                 forcePauseIntervals=(list(zip(sesh.homeRewardEnter_ts, sesh.homeRewardExit_ts)) +
                                                      list(zip(sesh.awayRewardEnter_ts, sesh.awayRewardExit_ts))))

    if sesh.probePerformed:
        sesh.probeSmoothVel, sesh.probeExploreBoutStart_posIdx, sesh.probeExploreBoutEnd_posIdx, \
            sesh.probeExploreBoutLensSecs, sesh.probeBoutCategory, sesh.probeBoutLabel = \
            getExplorationCategories(sesh.probePos_ts, sesh.probeVelCmPerSRaw,
                                     sesh.probeNearestWells, sesh.importOptions)


# def getExcursions(nearestWells: ArrayLike, ts: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
#     excursionCategory = np.array([BTSession.EXCURSION_STATE_ON_WALL if onWall(
#         w) else BTSession.EXCURSION_STATE_OFF_WALL for w in nearestWells])
#     excursionStarts = np.where(
#         np.diff((excursionCategory == BTSession.EXCURSION_STATE_OFF_WALL).astype(int)) == 1)[0] + 1
#     if excursionCategory[0] == BTSession.EXCURSION_STATE_OFF_WALL:
#         excursionStarts = np.insert(excursionStarts, 0, 0)
#     excursionEnds = np.where(
#         np.diff((excursionCategory == BTSession.EXCURSION_STATE_OFF_WALL).astype(int)) == -1)[0] + 1
#     if excursionCategory[-1] == BTSession.EXCURSION_STATE_OFF_WALL:
#         excursionEnds = np.append(excursionEnds, len(excursionCategory))
#     t = np.array(ts + [ts[-1]])
#     excursionLensSecs = (t[excursionEnds - 1] - t[excursionStarts]) / TRODES_SAMPLING_RATE

#     return excursionCategory, excursionStarts, excursionEnds, excursionLensSecs

def getExcursions(xs: ArrayLike, ys: ArrayLike, ts: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    excursionStartCategory = np.array([BTSession.EXCURSION_STATE_ON_WALL if posOnWall(
        (x, y)) else BTSession.EXCURSION_STATE_OFF_WALL for x, y in zip(xs, ys)])
    excursionStopCategory = np.array([BTSession.EXCURSION_STATE_ON_WALL if posOnWall(
        (x, y), buffer=-0.25) else BTSession.EXCURSION_STATE_OFF_WALL for x, y in zip(xs, ys)])
    excursionStarts = np.where(
        np.diff((excursionStartCategory == BTSession.EXCURSION_STATE_OFF_WALL).astype(int)) == 1)[0] + 1
    if excursionStartCategory[0] == BTSession.EXCURSION_STATE_OFF_WALL:
        excursionStarts = np.insert(excursionStarts, 0, 0)

    excursionEnds = np.empty_like(excursionStarts)
    # for each start, find the first index afterward for which excursionstopcategory is on wall
    for i, s in enumerate(excursionStarts):
        try:
            if len(np.where(excursionStopCategory[s:] == BTSession.EXCURSION_STATE_ON_WALL)[0]) == 0:
                excursionEnds[i] = len(excursionStopCategory)
            else:
                excursionEnds[i] = np.where(excursionStopCategory[s:] ==
                                            BTSession.EXCURSION_STATE_ON_WALL)[0][0] + s
        except IndexError as ie:
            print(ie)
            print(i, s)
            print(excursionStarts)
            print(np.where(excursionStopCategory[s:] == BTSession.EXCURSION_STATE_ON_WALL)[0])
            print(len(np.where(excursionStopCategory[s:] == BTSession.EXCURSION_STATE_ON_WALL)[0]))
            raise ie

    # Now if two excursion starts have the same end, only keep the first one
    keepFlag = np.diff(excursionEnds, prepend=-1) != 0
    excursionStarts = excursionStarts[keepFlag]
    excursionEnds = excursionEnds[keepFlag]

    # if excursionStopCategory[-1] == BTSession.EXCURSION_STATE_OFF_WALL:
    # excursionEnds = np.append(excursionEnds, len(excursionStopCategory))
    t = np.array(ts + [ts[-1]])
    excursionLensSecs = (t[excursionEnds - 1] - t[excursionStarts]) / TRODES_SAMPLING_RATE
    excursionCategory = np.full_like(xs, BTSession.EXCURSION_STATE_ON_WALL)
    for s, e in zip(excursionStarts, excursionEnds):
        excursionCategory[s:e] = BTSession.EXCURSION_STATE_OFF_WALL

    return excursionCategory, excursionStarts, excursionEnds, excursionLensSecs


@TimeThisFunction
def posCalcExcursions(sesh: BTSession) -> None:
    sesh.btExcursionCategory, sesh.btExcursionStart_posIdx, sesh.btExcursionEnd_posIdx, \
        sesh.btExcursionLensSecs = getExcursions(sesh.btPosXs, sesh.btPosYs, sesh.btPos_ts)

    if sesh.probePerformed:
        sesh.probeExcursionCategory, sesh.probeExcursionStart_posIdx, sesh.probeExcursionEnd_posIdx, \
            sesh.probeExcursionLensSecs = getExcursions(
                sesh.probePosXs, sesh.probePosYs, sesh.probePos_ts)

    # sesh.btExcursionCategory, sesh.btExcursionStart_posIdx, sesh.btExcursionEnd_posIdx, \
    #     sesh.btExcursionLensSecs = getExcursions(sesh.btNearestWells, sesh.btPos_ts)

    # if sesh.probePerformed:
    #     sesh.probeExcursionCategory, sesh.probeExcursionStart_posIdx, sesh.probeExcursionEnd_posIdx, \
    #         sesh.probeExcursionLensSecs = getExcursions(
    #             sesh.probeNearestWells, sesh.probePos_ts)


def runPositionAnalyses(sesh: BTSession) -> None:
    if sesh.lastAwayWell is None:
        sesh.numAwayFound = 0
    else:
        sesh.numAwayFound = next((i for i in range(
            len(sesh.awayWells)) if sesh.awayWells[i] == sesh.lastAwayWell), -1) + 1
    sesh.visitedAwayWells = sesh.awayWells[0:sesh.numAwayFound]
    # print(sesh.lastAwayWell)
    sesh.numHomeFound = sesh.numAwayFound
    if sesh.endedOnHome:
        sesh.numHomeFound += 1

    if not sesh.hasPositionData:
        print("\tCan't run analyses without any data!")
        return

    if sesh.hasRewardFindTimes:
        assert sesh.numAwayFound == len(sesh.awayRewardEnter_ts)
        assert sesh.numHomeFound == len(sesh.homeRewardEnter_ts)

    posCalcVelocity(sesh)
    posCalcEntryExitTimes(sesh)

    # ===================================
    # truncate path when rat was recalled (as he's running to be picked up) based on visit time of marked well
    # Note need to check sessions that ended near 7
    # ===================================
    # btRecall_posIdx
    if sesh.btEndedAtWell is None:
        assert sesh.lastAwayWell == sesh.awayWells[-1] and sesh.endedOnHome
        sesh.btEndedAtWell = sesh.homeWell

    sesh.btRecall_posIdx = sesh.btWellExitTimes_posIdx[sesh.btEndedAtWell][-1]

    if sesh.probePerformed:
        sesh.probeRecall_posIdx = sesh.probeWellExitTimes_posIdx[sesh.probeEndedAtWell][-1]

    posCalcCurvature(sesh)

    posCalcExplorationBouts(sesh)
    posCalcExcursions(sesh)
    # posCalcRewardCategories(sesh)


def extractAndSave(configName: str, importOptions: ImportOptions) -> BTData:
    numExcluded = 0
    numExtracted = 0

    print("=========================================================")
    print(f"Extracting data for animal {configName}")
    print("=========================================================")
    # animalInfo = getInfoForAnimal(animalName)
    loadInfo = getLoadInfo(configName)
    print(
        f"\tdataDir = {loadInfo.data_dir}\n\toutputDir = {loadInfo.output_dir}")
    if not os.path.exists(loadInfo.output_dir):
        os.mkdir(loadInfo.output_dir)

    sessionDirs, prevSessionDirs = getSessionDirs(loadInfo, importOptions)
    dirListStr = "".join(
        [f"\t{s}\t{ss}\n" for s, ss in zip(sessionDirs, prevSessionDirs)])
    print("Session dirs:")
    print(dirListStr)

    dataObj = BTData()
    dataObj.importOptions = importOptions

    lastDateStr = ""
    lastPrevDateStr = ""
    sessionNumber = None
    prevSessionNumber = None
    infoProblemSessions = []
    posProblemSessions = []
    for seshi, (seshDir, prevSeshDir) in enumerate(zip(sessionDirs, prevSessionDirs)):
        if importOptions.debugMode and \
            importOptions.debug_maxNumSessions is not None and \
                seshi >= importOptions.debug_maxNumSessions:
            break
        dateStr = seshDir.split('_')[0][-8:]
        if dateStr == lastDateStr:
            sessionNumber += 1
        else:
            lastDateStr = dateStr
            sessionNumber = 1

        if prevSeshDir is not None:
            prevDateStr = prevSeshDir.split('_')[0][-8:]
            if prevDateStr == lastPrevDateStr:
                prevSessionNumber += 1
            else:
                lastPrevDateStr = prevDateStr
                prevSessionNumber = 1

        sesh = makeSessionObj(seshDir, prevSeshDir, sessionNumber, prevSessionNumber, loadInfo,
                              skipActivelinkLog=importOptions.skipActivelinkLog)
        sesh.animalName = loadInfo.animalName
        sesh.importOptions = dataclasses.replace(importOptions)
        print("\n=======================================")
        print(f"Starting session {sesh.name}:")
        print(f"\tfolder: {seshDir}:")
        print(f"\tinfo file name: {sesh.infoFileName}:")
        print(f"\tprev sesh info file name: {sesh.prevInfoFileName }:")
        print(f"\tactivelink log file name: {sesh.activelinkLogFileName }:")
        if seshDir in loadInfo.excluded_sessions:
            print(seshDir, " excluded session, skipping")
            numExcluded += 1
            continue
        if "".join(os.path.basename(sesh.infoFileName).split(".")[0:-1]) in loadInfo.excluded_sessions:
            print(seshDir, " excluded session, skipping")
            numExcluded += 1
            continue

        if sesh.importOptions.confirmAfterEachSession:
            res = input("Press enter to continue, type s to skip, q to quit")
            if res == "q":
                return
            elif res == "s":
                continue

        # print("Parsing info files")
        try:
            parseInfoFiles(sesh)
        except Exception as e:
            print("Error parsing info file, skipping session", sesh.name)
            infoProblemSessions.append((sesh.name, e))
            continue
        parseActivelinkLog(sesh)
        # print(sesh.loggedDetections_ts)
        # print(sesh.loggedStats)
        # try:
        # except Exception as e:
        #     print("Error parsing activelink log file, skipping session", sesh.name)
        #     infoProblemSessions.append((sesh.name, e))
        #     continue

        # print("Loading position data")
        loadPositionData(sesh)
        if sesh.isNoInterruption:
            sesh.importOptions.skipLFP = True
        if sesh.importOptions.skipLFP:
            print("!!!!!\tSKIPPING LFP\t!!!!!")
            lfpData = None
            baselineLfpData = None
        else:
            # print("Loading LFP")
            lfpData, baselineLfpData = loadLFPData(sesh)

        if sesh.importOptions.justExtractData:
            print("Just extracting the data, continuing to next session...\n")
            numExtracted += 1
            continue

        if not sesh.importOptions.skipLFP:
            # print("Analyzing LFP")
            runLFPAnalyses(sesh, lfpData, baselineLfpData)
        # print("Analyzing position")
        try:
            runPositionAnalyses(sesh)
        except Exception as e:
            print("Error running position analyses, skipping session", sesh.name)
            posProblemSessions.append((sesh.name, e))
            continue

        # print("Running sanity checks")
        runSanityChecks(sesh, lfpData, baselineLfpData)

        # print(f"Done with session {sesh.name}")
        dataObj.allSessions.append(sesh)

    if importOptions.justExtractData:
        print(
            f"Extracted data from {numExtracted} sessions, excluded {numExcluded}. Not analyzing or saving")
        if len(infoProblemSessions) > 0:
            print("Had info problems with the following sessions:")
            for name, e in infoProblemSessions:
                print(name, e)
        if len(posProblemSessions) > 0:
            print("Had position problems with the following sessions:")
            for name, e in posProblemSessions:
                print(name, e)
        return

    if not any([s.conditionGroup is not None for s in dataObj.allSessions]):
        for si, sesh in enumerate(dataObj.allSessions):
            sesh.conditionGroup = sesh.animalName + "-" + str(si)

    # save all sessions to disk
    if importOptions.debugMode and importOptions.debug_dontSave:
        print("Not saving because in debug mode")
        print("Would've saved the following sessions:")
        for sesh in dataObj.allSessions:
            print(sesh.name)
    else:
        if importOptions.debugMode:
            outputFname = os.path.join(loadInfo.output_dir,
                                       loadInfo.out_filename + ".debug.dat")
        else:
            outputFname = os.path.join(loadInfo.output_dir, loadInfo.out_filename)
        print("Saving to file: {}".format(outputFname))
        dataObj.saveToFile(outputFname)
        print("Saved sessions:")
        for sesh in dataObj.allSessions:
            print(sesh.name)

    if len(infoProblemSessions) > 0:
        print("Had info problems with the following sessions:")
        for name, e in infoProblemSessions:
            print(name, e)
    if len(posProblemSessions) > 0:
        print("Had position problems with the following sessions:")
        for name, e in posProblemSessions:
            print(name, e)

    print(f"\tposCalcVelocity: {posCalcVelocity.totalTime}")
    print(f"\tposCalcEntryExitTimes: {posCalcEntryExitTimes.totalTime}")
    print(f"\tposCalcCurvature: {posCalcCurvature.totalTime}")
    print(f"\tposCalcExplorationBouts: {posCalcExplorationBouts.totalTime}")
    print(f"\tposCalcExcursions: {posCalcExcursions.totalTime}")

    return dataObj


def parseCommandLineArgs():
    animalNames = parseCmdLineAnimalNames(default=["B17"])
    importOptions = parseCmdLineImportOptions()
    return animalNames, importOptions


def main():
    animalNames, importOptions = parseCommandLineArgs()

    for animalName in animalNames:
        importOptions.skipUSB = animalName == "Martin" or True
        importOptions.skipActivelinkLog = animalName == "Martin" or True
        extractAndSave(animalName, importOptions)


if __name__ == "__main__":
    main()
