#! /usr/bin/python
# -*- coding: utf-8 -*-

#
# tkinter example for VLC Python bindings
# Copyright (C) 2015 the VideoLAN team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
#
"""Adapted from a simple example for VLC python bindings using tkinter by
Patrick Fay. https://github.com/oaubert/python-vlc/blob/master/examples/tkvlc.py
"""

import argparse as ap
import json
import numpy as np
import os
import pathlib
import platform
from subprocess import PIPE, run
from threading import Thread, Event
import time
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory, askopenfilename
import vlc

parser = ap.ArgumentParser()
parser.add_argument('--defaultdatasourcepath', '-d', default='C:/Users/Public/fra-gctd-project/Data_Sources/ramsey_nj')
args = parser.parse_args()


class ttkTimer(Thread):
  """a class serving same function as wxTimer... but there may be better ways to do this
  """

  def __init__(self, callback, tick):
    Thread.__init__(self)
    self.callback = callback
    self.stopFlag = Event()
    self.tick = tick
    self.iters = 0

  def run(self):
    while not self.stopFlag.wait(self.tick):
      self.iters += 1
      self.callback()

  def stop(self):
    self.stopFlag.set()

  def get(self):
    return self.iters


class Player(tk.Frame):
  """The main window has to deal with events.
  """

  def __init__(self, parent, title=None):
    tk.Frame.__init__(self, parent)

    self.parent = parent

    if title == None:
      title = "tk_vlc"
    self.parent.title(title)

    # Menu Bar
    #   File Menu
    menubar = tk.Menu(self.parent)
    self.parent.config(menu=menubar)

    fileMenu = tk.Menu(menubar)
    fileMenu.add_command(label="Open File", underline=0, command=self.OnOpenFile)
    fileMenu.add_command(label="Open Directory", underline=0, command=self.OnOpenDirectory)
    fileMenu.add_command(label="Exit", underline=1, command=_quit)
    menubar.add_cascade(label="File", menu=fileMenu)

    # The second panel holds controls
    self.player = None

    videolabelpanel = ttk.Frame(self.parent)
    self.videolabelvar = tk.StringVar()
    self.videolabel = ttk.Label(
      master=videolabelpanel, textvariable=self.videolabelvar).pack(
      side=tk.TOP)
    videolabelpanel.pack(side=tk.TOP)

    self.videopanel = ttk.Frame(self.parent)
    self.canvas = tk.Canvas(master=self.videopanel).pack(
      fill=tk.BOTH, expand=1)
    self.videopanel.pack(fill=tk.BOTH, expand=1)

    ctrlpanel = ttk.Frame(self.parent)

    prev = ttk.Button(ctrlpanel, text="Prev", command=self.OnPrev)
    play = ttk.Button(ctrlpanel, text="Play", command=self.OnPlay)
    replay = ttk.Button(ctrlpanel, text="Replay", command=self.OnReplay)
    next = ttk.Button(ctrlpanel, text="Next", command=self.OnNext)
    save = ttk.Button(ctrlpanel, text="Save", command=self.OnSave)
    load = ttk.Button(ctrlpanel, text="Load", command=self.OnLoad)

    prev.pack(side=tk.LEFT)
    play.pack(side=tk.LEFT)
    replay.pack(side=tk.LEFT)
    next.pack(side=tk.LEFT)
    save.pack(side=tk.LEFT)
    load.pack(side=tk.LEFT)

    ctrlpanel.pack(side=tk.BOTTOM)

    ctrlpanel2 = ttk.Frame(self.parent)
    self.scale_var = tk.DoubleVar()
    self.timeslider_last_val = ""
    self.timeslider = tk.Scale(ctrlpanel2, variable=self.scale_var,
                               command=self.scale_sel,
                               from_=0, to=1000, orient=tk.HORIZONTAL,
                               length=500)
    self.timeslider.pack(side=tk.BOTTOM, fill=tk.X, expand=1)
    self.timeslider_last_update = time.time()
    ctrlpanel2.pack(side=tk.BOTTOM, fill=tk.X)

    # VLC player controls
    self.Instance = vlc.Instance()
    self.player = self.Instance.media_player_new()

    self.directory_path = None
    self.directory_child_filenames = []

    self.num_clips = None
    self.current_clip = None

    self.class_list = [
      'anomaly',
      'gates_up',
      'gates_down',
      'gates_ascending',
      'gates_descending',
      'gate_lights_flshng',
      'trn_adv_on_se_corr',
      'trn_adv_on_se_crsg',
      'trn_adv_on_ne_corr',
      'trn_adv_on_ne_crsg',
      'trn_adv_on_sw_corr',
      'trn_adv_on_sw_crsg',
      'trn_adv_on_nw_corr',
      'trn_adv_on_nw_crsg',
      'trn_rec_on_se_corr',
      'trn_rec_on_se_crsg',
      'trn_rec_on_ne_corr',
      'trn_rec_on_ne_crsg',
      'trn_rec_on_sw_corr',
      'trn_rec_on_sw_crsg',
      'trn_rec_on_nw_corr',
      'trn_rec_on_nw_crsg',
      'trn_std_on_se_corr',
      'trn_std_on_se_crsg',
      'trn_std_on_ne_corr',
      'trn_std_on_ne_crsg',
      'trn_std_on_sw_corr',
      'trn_std_on_sw_crsg',
      'trn_std_on_nw_corr',
      'trn_std_on_nw_crsg',
      'veh_adv_on_se_corr',
      'veh_adv_on_se_crsg',
      'veh_adv_on_se_strt',
      'veh_adv_on_ne_corr',
      'veh_adv_on_ne_crsg',
      'veh_adv_on_ne_strt',
      'veh_adv_on_sw_corr',
      'veh_adv_on_sw_crsg',
      'veh_adv_on_sw_strt',
      'veh_adv_on_nw_corr',
      'veh_adv_on_nw_crsg',
      'veh_adv_on_nw_strt',
      'veh_rec_on_se_corr',
      'veh_rec_on_se_crsg',
      'veh_rec_on_se_strt',
      'veh_rec_on_ne_corr',
      'veh_rec_on_ne_crsg',
      'veh_rec_on_ne_strt',
      'veh_rec_on_sw_corr',
      'veh_rec_on_sw_crsg',
      'veh_rec_on_sw_strt',
      'veh_rec_on_nw_corr',
      'veh_rec_on_nw_crsg',
      'veh_rec_on_nw_strt',
      'veh_std_on_se_corr',
      'veh_std_on_se_crsg',
      'veh_std_on_se_strt',
      'veh_std_on_ne_corr',
      'veh_std_on_ne_crsg',
      'veh_std_on_ne_strt',
      'veh_std_on_sw_corr',
      'veh_std_on_sw_crsg',
      'veh_std_on_sw_strt',
      'veh_std_on_nw_corr',
      'veh_std_on_nw_crsg',
      'veh_std_on_nw_strt',
      'cyc_adv_on_se_corr',
      'cyc_adv_on_se_crsg',
      'cyc_adv_on_se_strt',
      'cyc_adv_on_se_sdwk',
      'cyc_adv_on_ne_corr',
      'cyc_adv_on_ne_crsg',
      'cyc_adv_on_ne_strt',
      'cyc_adv_on_ne_sdwk',
      'cyc_adv_on_sw_corr',
      'cyc_adv_on_sw_crsg',
      'cyc_adv_on_sw_strt',
      'cyc_adv_on_sw_sdwk',
      'cyc_adv_on_nw_corr',
      'cyc_adv_on_nw_crsg',
      'cyc_adv_on_nw_strt',
      'cyc_adv_on_nw_sdwk',
      'cyc_rec_on_se_corr',
      'cyc_rec_on_se_crsg',
      'cyc_rec_on_se_strt',
      'cyc_rec_on_se_sdwk',
      'cyc_rec_on_ne_corr',
      'cyc_rec_on_ne_crsg',
      'cyc_rec_on_ne_strt',
      'cyc_rec_on_ne_sdwk',
      'cyc_rec_on_sw_corr',
      'cyc_rec_on_sw_crsg',
      'cyc_rec_on_sw_strt',
      'cyc_rec_on_sw_sdwk',
      'cyc_rec_on_nw_corr',
      'cyc_rec_on_nw_crsg',
      'cyc_rec_on_nw_strt',
      'cyc_rec_on_nw_sdwk',
      'cyc_std_on_se_corr',
      'cyc_std_on_se_crsg',
      'cyc_std_on_se_strt',
      'cyc_std_on_se_sdwk',
      'cyc_std_on_ne_corr',
      'cyc_std_on_ne_crsg',
      'cyc_std_on_ne_strt',
      'cyc_std_on_ne_sdwk',
      'cyc_std_on_sw_corr',
      'cyc_std_on_sw_crsg',
      'cyc_std_on_sw_strt',
      'cyc_std_on_sw_sdwk',
      'cyc_std_on_nw_corr',
      'cyc_std_on_nw_crsg',
      'cyc_std_on_nw_strt',
      'cyc_std_on_nw_sdwk',
      'cyc_arnd_se_ped_gt',
      'cyc_arnd_ne_ped_gt',
      'cyc_arnd_ne_veh_gt',
      'cyc_arnd_sw_ped_gt',
      'cyc_arnd_sw_veh_gt',
      'cyc_arnd_nw_ped_gt',
      'cyc_over_se_ped_gt',
      'cyc_over_ne_ped_gt',
      'cyc_over_ne_veh_gt',
      'cyc_over_sw_ped_gt',
      'cyc_over_sw_veh_gt',
      'cyc_over_nw_ped_gt',
      'cyc_undr_se_ped_gt',
      'cyc_undr_ne_ped_gt',
      'cyc_undr_ne_veh_gt',
      'cyc_undr_sw_ped_gt',
      'cyc_undr_sw_veh_gt',
      'cyc_undr_nw_ped_gt',
      'ped_adv_on_se_corr',
      'ped_adv_on_se_crsg',
      'ped_adv_on_se_strt',
      'ped_adv_on_se_sdwk',
      'ped_adv_on_ne_corr',
      'ped_adv_on_ne_crsg',
      'ped_adv_on_ne_strt',
      'ped_adv_on_ne_sdwk',
      'ped_adv_on_ne_ptfm',
      'ped_adv_on_sw_corr',
      'ped_adv_on_sw_crsg',
      'ped_adv_on_sw_strt',
      'ped_adv_on_sw_sdwk',
      'ped_adv_on_nw_corr',
      'ped_adv_on_nw_crsg',
      'ped_adv_on_nw_strt',
      'ped_adv_on_nw_sdwk',
      'ped_adv_on_nw_ptfm',
      'ped_rec_on_se_corr',
      'ped_rec_on_se_crsg',
      'ped_rec_on_se_strt',
      'ped_rec_on_se_sdwk',
      'ped_rec_on_ne_corr',
      'ped_rec_on_ne_crsg',
      'ped_rec_on_ne_strt',
      'ped_rec_on_ne_sdwk',
      'ped_rec_on_ne_ptfm',
      'ped_rec_on_sw_corr',
      'ped_rec_on_sw_crsg',
      'ped_rec_on_sw_strt',
      'ped_rec_on_sw_sdwk',
      'ped_rec_on_nw_corr',
      'ped_rec_on_nw_crsg',
      'ped_rec_on_nw_strt',
      'ped_rec_on_nw_sdwk',
      'ped_rec_on_nw_ptfm',
      'ped_std_on_se_corr',
      'ped_std_on_se_crsg',
      'ped_std_on_se_strt',
      'ped_std_on_se_sdwk',
      'ped_std_on_ne_corr',
      'ped_std_on_ne_crsg',
      'ped_std_on_ne_strt',
      'ped_std_on_ne_sdwk',
      'ped_std_on_ne_ptfm',
      'ped_std_on_sw_corr',
      'ped_std_on_sw_crsg',
      'ped_std_on_sw_strt',
      'ped_std_on_sw_sdwk',
      'ped_std_on_nw_corr',
      'ped_std_on_nw_crsg',
      'ped_std_on_nw_strt',
      'ped_std_on_nw_sdwk',
      'ped_std_on_nw_pftm',
      'ped_arnd_se_ped_gt',
      'ped_arnd_ne_ped_gt',
      'ped_arnd_ne_veh_gt',
      'ped_arnd_sw_ped_gt',
      'ped_arnd_sw_veh_gt',
      'ped_arnd_nw_ped_gt',
      'ped_over_se_ped_gt',
      'ped_over_ne_ped_gt',
      'ped_over_ne_veh_gt',
      'ped_over_sw_ped_gt',
      'ped_over_sw_veh_gt',
      'ped_over_nw_ped_gt',
      'ped_undr_se_ped_gt',
      'ped_undr_ne_ped_gt',
      'ped_undr_ne_veh_gt',
      'ped_undr_sw_ped_gt',
      'ped_undr_sw_veh_gt',
      'ped_undr_nw_ped_gt'
    ]

    self.default_data_source_path = args.defaultdatasourcepath

    try:
      ffmpeg_path = os.environ['FFMPEG_PATH']
    except KeyError:
      ffmpeg_path = '/usr/local/bin/ffmpeg'

      if not os.path.exists(ffmpeg_path):
        ffmpeg_path = '/usr/bin/ffmpeg'

    self.input_ffmpeg_command_prefix = [ffmpeg_path, '-i']

    self.input_ffmpeg_command_suffix = [
      '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', 'vfr',
      '-hide_banner', '-loglevel', '0', '-f', 'image2pipe', 'pipe:1']

    try:
      ffprobe_path = os.environ['FFPROBE_PATH']
    except KeyError:
      ffprobe_path = '/usr/local/bin/ffprobe'

      if not os.path.exists(ffprobe_path):
        ffprobe_path = '/usr/bin/ffprobe'

    self.input_ffprobe_command_prefix = [
      ffprobe_path, '-show_streams', '-print_format', 'json', '-loglevel',
      'warning']

    self.clip_string_len = 64 * 224 * 224 * 3

    self.buffer_scale = 2

    while self.buffer_scale < self.clip_string_len:
      self.buffer_scale *= 2

    self.labelpanel = ttk.Frame(self.parent)

    # create array of checkboxes
    for k, v in enumerate(self.class_list):
      ttk.Checkbutton(self.labelpanel, text=v, onvalue=1, offvalue=0).grid(
        column=k % 12, row=int(k / 12), sticky=tk.NW)

    self.labelpanel.pack(side=tk.BOTTOM)

    self.timer = ttkTimer(self.OnTimer, 1.0)
    self.timer.start()

    self.parent.update()

  def invoke_subprocess(self, command):
    completed_subprocess = run(
      command, stdout=PIPE, stderr=PIPE, timeout=60)

    if len(completed_subprocess.stderr) > 0:
      std_err = str(completed_subprocess.stderr, encoding='utf-8')

      raise Exception(std_err)

    return completed_subprocess.stdout

  def get_video_dimensions(self, video_file_path):
    command = self.input_ffprobe_command_prefix + [video_file_path]

    output = self.invoke_subprocess(command)

    json_map = json.loads(str(output, encoding='utf-8'))

    return int(json_map['streams'][0]['height']), \
           int(json_map['streams'][0]['width']), \
           int(json_map['streams'][0]['nb_frames'])

  def get_video_clip(self, video_file_path):
    command = self.input_ffmpeg_command_prefix + [video_file_path] \
              + self.input_ffmpeg_command_suffix

    output = self.invoke_subprocess(command)

    return output

  def OnExit(self, evt):
    """Closes the window.
    """
    self.Close()

  def OnOpenFile(self):
    """Pop up a new dialow window to choose a file, then play the selected file.
    """
    # if a file is already running, then stop it.
    self.OnStop()

    # Create a file dialog opened in the current PATH directory, where
    # you can display all kind of files, having as title "Choose a file".
    p = pathlib.Path(self.default_data_source_path)

    video_file_path = askopenfilename(
      initialdir=p, title="Choose a video clip", filetypes=(
        ("all files", "*.*"), ("mp4 files", "*.mp4"), ("avi files", "*.avi")))

    if os.path.isfile(video_file_path):
      self.directory_path = os.path.dirname(video_file_path)
      video_file_name = os.path.basename(video_file_path)

      self.directory_child_filenames = sorted(
        os.listdir(self.directory_path))

      self.num_clips = len(self.directory_child_filenames)

      self.current_clip = self.directory_child_filenames.index(video_file_name)

      self.DisplayClip()

  def OnOpenDirectory(self):
    """Pop up a new dialow window to choose a file, then play the selected file.
    """
    # if a file is already running, then stop it.
    self.OnStop()

    # Create a file dialog opened in the current PATH directory, where
    # you can display all kind of files, having as title "Choose a file".
    p = pathlib.Path(self.default_data_source_path)

    directory_path = askdirectory(
      initialdir=p, title='Choose a video clip parent directory')

    if os.path.isdir(directory_path):
      self.directory_path = directory_path

      self.directory_child_filenames = sorted(
        os.listdir(self.directory_path))

      self.num_clips = len(self.directory_child_filenames)

      self.current_clip = 0

      self.DisplayClip()

  def DisplayClip(self):
    # Creation
    media = self.Instance.media_new(os.path.join(
      self.directory_path, self.directory_child_filenames[self.current_clip]))
    self.player.set_media(media)
    self.videolabelvar.set(self.directory_child_filenames[self.current_clip])
    # set the window id where to render VLC's video output
    if platform.system() == 'Windows':
      self.player.set_hwnd(self.GetHandle())
    else:
      self.player.set_xwindow(self.GetHandle())  # this line messes up windows
    # TODO: this should be made cross-platform
    self.OnLoad()
    self.OnPlay()

  # TODO: add UI element to indacate the presence/absence of a label
  def OnLoad(self):
    """Skip to the next video in the directory.
    If a file was specified at __init__ rather than a directory, do nothing.
    """
    # check if there is a file to play, otherwise open a
    # tk.FileDialog to select a file
    if not self.player.get_media():
      self.OnOpenDirectory()
    else:
      # set the checkbutton state values equal to those in the label associated
      # with the currently visible video.
      try:
        save_path = os.path.join(
          os.path.dirname(self.directory_path), 'labels')
        label_array = np.load(os.path.join(save_path, os.path.splitext(
          self.directory_child_filenames[self.current_clip])[0] + '.npy'))
        print('loading label: {}'.format(label_array))
        checkbuttons = list(self.labelpanel.children.values())
        for i in range(len(checkbuttons)):
          checkbuttons[i].state(
            ('selected', '!alternate') if label_array[i] == 1
            else ('alternate', '!selected'))
      except:
        pass

  def OnSave(self):
    """Skip to the next video in the directory.
    If a file was specified at __init__ rather than a directory, do nothing.
    """
    # check if there is a file to play, otherwise open a
    # tk.FileDialog to select a file
    if not self.player.get_media():
      self.OnOpenDirectory()
    else:
      # save the selected labels, then advance the clip
      label_array = np.array(
        [checkbutton.instate(['selected']) for checkbutton
         in self.labelpanel.children.values()], dtype=np.uint8)
      print('saving label: {}'.format(label_array))
      save_path = os.path.join(os.path.dirname(self.directory_path), 'labels')
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      np.save(os.path.join(save_path, os.path.splitext(
        self.directory_child_filenames[self.current_clip])[0] + '.npy'),
              label_array)

      self.OnNext()

  def OnNext(self):
    """Skip to the next video in the directory.
    If a file was specified at __init__ rather than a directory, do nothing.
    """
    # check if there is a file to play, otherwise open a
    # tk.FileDialog to select a file
    if not self.player.get_media():
      self.OnOpenDirectory()
    else:
      if self.current_clip < self.num_clips - 1:
        self.current_clip += 1
      self.DisplayClip()
      self.OnPlay()

  def OnPrev(self):
    """Skip to the next video in the directory.
    If a file was specified at __init__ rather than a directory, do nothing.
    """
    # check if there is a file to play, otherwise open a
    # tk.FileDialog to select a file
    if not self.player.get_media():
      self.OnOpenDirectory()
    else:
      if self.current_clip > 0:
        self.current_clip -= 1
      self.DisplayClip()
      self.OnPlay()

  def OnPlay(self):
    """Toggle the status to Play/Pause.
    If no file is loaded, open the dialog window.
    """
    # check if there is a file to play, otherwise open a
    # tk.FileDialog to select a file
    if not self.player.get_media():
      self.OnOpenDirectory()
    else:
      # Try to launch the media, if this fails display an error message
      if self.player.play() == -1:
        self.errorDialog("Unable to play.")

  def GetHandle(self):
    return self.videopanel.winfo_id()

  # def OnPause(self, evt):
  def OnPause(self):
    """Pause the player.
    """
    self.player.pause()

  def OnReplay(self):
    """Stop the player.
    """
    self.player.stop()
    # reset the time slider
    self.timeslider.set(0)
    self.player.play()

  def OnStop(self):
    """Stop the player.
    """
    self.player.stop()
    # reset the time slider
    self.timeslider.set(0)

  def OnTimer(self):
    """Update the time slider according to the current movie time.
    """
    if self.player == None:
      return
    # since the self.player.get_length can change while playing,
    # re-set the timeslider to the correct range.
    length = self.player.get_length()
    dbl = length * 0.001
    self.timeslider.config(to=dbl)

    # update the time on the slider
    tyme = self.player.get_time()
    if tyme == -1:
      tyme = 0
    dbl = tyme * 0.001
    self.timeslider_last_val = ("%.0f" % dbl) + ".0"
    # don't want to programatically change slider while user is messing with it.
    # wait 2 seconds after user lets go of slider
    if time.time() > (self.timeslider_last_update + 2.0):
      self.timeslider.set(dbl)

  def scale_sel(self, evt):
    if self.player == None:
      return
    nval = self.scale_var.get()
    sval = str(nval)
    # this is a hack. The timer updates the time slider.
    # This change causes this rtn (the 'slider has changed' rtn) to be invoked.
    # I can't tell the difference between when the user has manually moved the slider and when
    # the timer changed the slider. But when the user moves the slider tkinter only notifies
    # this rtn about once per second and when the slider has quit moving.
    # Also, the tkinter notification value has no fractional seconds.
    # The timer update rtn saves off the last update value (rounded to integer seconds) in timeslider_last_val
    # if the notification time (sval) is the same as the last saved time timeslider_last_val then
    # we know that this notification is due to the timer changing the slider.
    # otherwise the notification is due to the user changing the slider.
    # if the user is changing the slider then I have the timer routine wait for at least
    # 2 seconds before it starts updating the slider again (so the timer doesn't start fighting with the
    # user)

    if self.timeslider_last_val != sval:
      self.timeslider_last_update = time.time()
      mval = "%.0f" % (nval * 1000)
      self.player.set_time(int(mval))  # expects milliseconds

  def errorDialog(self, errormessage):
    """Display a simple error dialog.
    """
    tk.TkMessageBox.showerror(self, 'Error', errormessage)


def tk_get_root():
  if not hasattr(tk_get_root, "root"):  # (1)
    tk_get_root.root = tk.Tk()  # initialization call is inside the function
  return tk_get_root.root


def _quit():
  print("_quit: bye")
  root = tk_get_root()
  root.quit()  # stops mainloop
  root.destroy()  # this is necessary on Windows to prevent
  # Fatal Python Error: PyEval_RestoreThread: NULL tstate
  os._exit(1)


if __name__ == "__main__":
  # Create a tk.App(), which handles the windowing system event loop
  root = tk_get_root()
  root.protocol("WM_DELETE_WINDOW", _quit)

  player = Player(root, title="GCTD Video Annotation Tool")
  # show the player window centred and run the application
  root.mainloop()
