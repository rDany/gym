"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
import re
import os
from gym import spaces
from gym.utils import seeding
import numpy as np
import random as rd

logger = logging.getLogger(__name__)


## MIDI
def open_midi(midigram_filepath, shift_pitch=0):
    ## Midigram collumns
    # 1: On (MIDI pulse units)
    # 2: Off (MIDI pulse units)
    # 3: Track number
    # 4: Channel number
    # 5: Midi pitch -> 60 -> c4 / 61 -> c#4
    # 6: Midi velocity

    midi_pulse_per_second  = 480.299
    fps = 30
    fps = 11.7 #TODO
    margin = 1
    keyboard_size = 88
    max_frames = 10000

    midi_pulse_per_frame = int(midi_pulse_per_second / fps)

    with open (midigram_filepath, "r", encoding="ISO-8859-1") as midigram_file:
        data=midigram_file.readlines()

    last_frame = 0
    lower_pitch = None
    midi_frames = {}
    for data_line in data[:-1]:
        line_regex = re.search(r'^(?P<ontime>\d+)\s(?P<offtime>\d+)\s(?P<track>\d+)\s(?P<channel>\d+)\s(?P<pitch>\d+)\s(?P<velocity>\d+)', data_line)
        if line_regex == None:
            continue
        ontime = int(int(line_regex.group("ontime"))/midi_pulse_per_frame)
        offtime = int(int(line_regex.group("offtime"))/midi_pulse_per_frame)
        track = int(line_regex.group("track"))
        channel = int(line_regex.group("channel"))
        pitch = int(line_regex.group("pitch"))
        velocity = int(line_regex.group("velocity"))

        if not ontime in midi_frames:
            midi_frames[ontime] = []
        midi_frames[ontime].append({
            "ontime": ontime,
            "offtime": offtime,
            "track": track,
            "channel": channel,
            "pitch": pitch,
            "velocity": velocity,
        })
        if offtime > last_frame:
            last_frame = offtime
        if lower_pitch == None:
            lower_pitch = pitch
        if pitch < lower_pitch:
            lower_pitch = pitch

    music = []
    for frame_number in range(0, last_frame+1):
        if frame_number > max_frames:
            break
        notes = [0 for _ in range(0, keyboard_size)]
        music.append(notes)

    for mf in midi_frames:
        for mf_note in midi_frames[mf]:
            # Normalize pitch on the keyboard
            mf_normalized_pitch = mf_note['pitch']-shift_pitch
            #mf_normalized_pitch = mf_note['pitch'] # Disabling normalization
            if mf_normalized_pitch >= keyboard_size:
                print ("Skipping note, out of keyboard range")
                continue
            #if mf_note['track'] != 1:
            #    print ("Skipping note, out of track")
            #    continue
            # Fill ontime to offtime values
            for f in range(mf_note['ontime'], mf_note['offtime']):
                music[f][mf_normalized_pitch] = 1.0

    notes = [0 for _ in range(0, keyboard_size)]
    music = [notes for _ in range(0, margin)] + music + [notes for _ in range(0, margin)]
    return music
## MIDI


class PianoEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }


    def __init__(self):
        #self.env_name = name
        self.frames = open_midi(os.path.join(os.path.dirname(__file__), 'assets/hummelflug.midigram'), 30)
        #self.frames = None
        #print (len(self.frames))

        self.key_count = 88
        self.fingers_count = 10
        actions_count = 6 # Actions except "stay still"
        self.frame_count = len(self.frames)
        #self.frame_count = 0
        self.randomize_start = False
        self.failed_frames_threshold = 30000

        #self.action_space = ActionSpace(self.fingers_count)
        self.action_space = spaces.Discrete((self.fingers_count*actions_count)+1)
        self.observation_space = spaces.Discrete((self.fingers_count*self.key_count)+self.key_count)

        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.action_history = []

        #print ("Generating actions")
        #step_sizes = [0, 1, -1, 2, -2, 3, -3]
        #self.finger_actions = []
        #print ("Actions generated")
        #self.reset()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _step(self, action):
        step_sizes = [1, -1, 2, -2, 3, -3]

        #finger_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # Thumb possible positions, being 4 the center (middle finger)
        #finger_2 = [0, 1, 2, 3, 4, 5] # Index possible positions, being 0 the center (middle finger)
        #finger_4 = [0, 1, 2, 3] # Ring possible positions, being 3 the center (middle finger)
        #finger_5 = [0, 1, 2, 3, 4, 5] # Pinky possible positions, being 5 the center (middle finger)

        # 7*14*6*4*6 = 14,112
        # 7*14*6*4*6 = 14,112
        # 14.112*14.112 = 199,148,544

        # 14*6*4*6 ^ 2 = 4,064,256

        # 7*7*7*7*7 = 16,807

        # 7*5*5*5*5 = 4,375*4,375 = 19,140,625
        # 5*5*5*5*5 = 3,125*3,125 = 9,765,625

        self.action_history.append(action)

        if action == 60:
            step_index = 0
            finger = None
        else:
            step_index = int(action/self.fingers_count)
            finger = action-(step_index*self.fingers_count)

        # Before applying actions, save current position
        previous_fingers_position = list(self.current_finger_position)

        # Apply Middle finger movements
        #for n_finger in range(0, self.fingers_count):
        #    self.current_finger_position[2] += step_sizes[step_index]
        #for n_finger in range(0, self.fingers_count):
        #    self.current_finger_position[7] += step_sizes[step_index]

        if finger != 2 and finger != 7 and finger != None:
            #print ("Action: ", action)
            #print ("Fingers count: ", self.fingers_count)
            #print ("Finger: ", finger)
            #print ("Step index: ", step_index)
            #print ()
            self.current_finger_position[finger] += step_sizes[step_index]
        if finger == 2: # Fingers 0, 1, 3, 4 constrained to 2
            for fn in range(0, 5):
                self.current_finger_position[fn] += step_sizes[step_index]
        elif finger == 7: # Fingers 5, 6, 8, 9 constrained to 7
            for fn in range(5, 10):
                self.current_finger_position[fn] += step_sizes[step_index]

        pinky_min = 6  # Left
        pinky_max = 2  # Right
        ring_min = 4   # Left
        ring_max = 1   # Right
        index_min = 6  # Right
        index_max = 1  # Left
        thumb_min = 10 # Right
        thumb_max = -2 #4  # Left

        if finger == 0:
            # Apply constrains Left hand fingers (constrained to left Middle)
            if self.current_finger_position[finger] < self.current_finger_position[2]-pinky_min:
                # If left Pinky is more than 5 keys away to the left, constrain
                self.current_finger_position[finger] = self.current_finger_position[2]-pinky_min
            elif self.current_finger_position[finger] > self.current_finger_position[2]-pinky_max:
                # If left Pinky is more than 0 keys away to the right, constrain
                self.current_finger_position[finger] = self.current_finger_position[2]-pinky_max
        elif finger == 1:
            if self.current_finger_position[finger] < self.current_finger_position[2]-ring_min:
                # If left Ring is more than 3 keys away to the left, constrain
                self.current_finger_position[finger] = self.current_finger_position[2]-ring_min
            elif self.current_finger_position[finger] > self.current_finger_position[2]-ring_max:
                # If left Ring is more than 0 keys away to the right, constrain
                self.current_finger_position[finger] = self.current_finger_position[2]-ring_max
        elif finger == 3: # We skip finger 2, since it is the Left Middle finger
            if self.current_finger_position[finger] > self.current_finger_position[2]+index_min:
                # If left Index is more than 5 keys away to the right, constrain
                self.current_finger_position[finger] = self.current_finger_position[2]+index_min
            elif self.current_finger_position[finger] < self.current_finger_position[2]+index_max:
                # If left Index is more than 0 keys away to the left, constrain
                self.current_finger_position[finger] = self.current_finger_position[2]+index_max
        elif finger == 4:
            if self.current_finger_position[finger] > self.current_finger_position[2]+thumb_min:
                # If left Thumb is more than X keys away to the right, constrain
                self.current_finger_position[finger] = self.current_finger_position[2]+thumb_min
            elif self.current_finger_position[finger] < self.current_finger_position[2]-thumb_max:
                # If left Thumb is more than X keys away to the left, constrain
                self.current_finger_position[finger] = self.current_finger_position[2]-thumb_max
        if finger == 5:
            # Apply constrains Right hand fingers (constrained to right Middle)
            if self.current_finger_position[finger] < self.current_finger_position[7]-thumb_min:
                # If right Thumb is more than 9 keys away to the left, constrain
                self.current_finger_position[finger] = self.current_finger_position[7]-thumb_min
            elif self.current_finger_position[finger] > self.current_finger_position[7]+thumb_max:
                # If right Thumb is more than 4 keys away to the right, constrain
                self.current_finger_position[finger] = self.current_finger_position[7]+thumb_max
        elif finger == 6:
            if self.current_finger_position[finger] < self.current_finger_position[7]-index_min:
                # If right Index is more than 5 keys away to the left, constrain
                self.current_finger_position[finger] = self.current_finger_position[7]-index_min
            elif self.current_finger_position[finger] > self.current_finger_position[7]-index_max:
                # If right Index is more than 0 keys away to the right, constrain
                self.current_finger_position[finger] = self.current_finger_position[7]-index_max
        elif finger == 8: # We skip finger 7, since it is the Right Middle finger
            if self.current_finger_position[finger] < self.current_finger_position[7]+ring_max:
                # If right Ring is more than 0 keys away to the left, constrain
                self.current_finger_position[finger] = self.current_finger_position[7]+ring_max
            elif self.current_finger_position[finger] > self.current_finger_position[7]+ring_min:
                # If right Ring is more than 3 keys away to the right, constrain
                self.current_finger_position[finger] = self.current_finger_position[7]+ring_min
        elif finger == 9:
            if self.current_finger_position[finger] < self.current_finger_position[7]+pinky_max:
                # If right Pinky is more than 0 keys away to the left, constrain
                self.current_finger_position[finger] = self.current_finger_position[7]+pinky_max
            elif self.current_finger_position[finger] > self.current_finger_position[7]+pinky_min:
                # If right Pinky is more than 5 keys away to the right, constrain
                self.current_finger_position[finger] = self.current_finger_position[7]+pinky_min

        # If a finger falls outside the keyboard, reset positions
        for finger_number in range(0, self.fingers_count):
            if self.current_finger_position[finger_number] < 0:
                self.current_finger_position = list(previous_fingers_position)
                break
            elif self.current_finger_position[finger_number] >= self.key_count:
                self.current_finger_position = list(previous_fingers_position)
                #print ("Finger position > Key Count")
                #print (previous_fingers_position)
                break

        if 0:
            default_reward = 0.001
            # If two fingers are in the same place, decrease reward
            for f in range(0, self.fingers_count):
                for ff in range(0, self.fingers_count):
                    if f != ff:
                        if self.current_finger_position[f] == self.current_finger_position[ff]:
                            default_reward -= 0.000001

            # If a finger is on the right of a finger with bigger number, decrease reward
            for f in range(0, self.fingers_count):
                for ff in range(f, self.fingers_count):
                    if f != ff:
                        if self.current_finger_position[f] > self.current_finger_position[ff]:
                            default_reward -= 0.000001

            # If a finger is on the left of a finger with smaller number, decrease reward
            for f in range(0, self.fingers_count):
                for ff in range(0, f):
                    if f != ff:
                        if self.current_finger_position[f] < self.current_finger_position[ff]:
                            default_reward -= 0.000001

            if default_reward < 0:
                default_reward = 0.000001

        # One hot representation of keys with fingers on it
        one_hot = [0 for _ in range(0, self.key_count*self.fingers_count)]
        for index, finger_key in enumerate(self.current_finger_position):
            #if finger_key < 0 or finger_key >= self.key_count:
            #    continue # Skip if the finger is outside the keyboard
            try:
                # 88 + (9*88)
                # finger_key = 88
                # index = 9
                # self.key_count = 88
                one_hot_index = finger_key+(index*self.key_count)
                one_hot[one_hot_index] = 1
            except IndexError:
                print (finger_key)
                print (index)
                print (self.key_count)
                print (finger_key+(index*self.key_count))
                raise

        done = False
        must_reset = False
        reward = 0.0

        # If no frames left, done
        if self.current_frame >= len(self.frames)-1-1:
            must_reset = True
            self.state = np.array(one_hot + self.frames[self.current_frame-1] + self.frames[self.current_frame-1])
        else:
            self.state = np.array(one_hot + self.frames[self.current_frame] + self.frames[self.current_frame + 1])

        if False:
            if not must_reset:
                # If "no movement" action was the last 100 actions, done
                frozen = True
                for act in action_history[:-100]:
                    if act!=61:
                        frozen = False
                if frozen:
                    must_reset = True
                    reward = -1.0

        failed = False
        achieved = False
        if not must_reset:
            for key, key_val in enumerate(self.frames[self.current_frame]):
                if key_val == 1.0 and key not in self.current_finger_position:
                    # If a key should be pressed but is not, increase failed_frames counter
                    failed = True
                    self.failed_frames += 1
                    if self.failed_frames > self.failed_frames_threshold:
                        # If failed_frames counter > threshold, done
                        must_reset = True
                elif key_val == 1.0 and key in self.current_finger_position:
                    # If a key should be pressed and it is, reward
                    #reward += 0.05
                    achieved = True
            if achieved:
                reward = 1.0
            if failed:
                reward = -1.0

        if not failed:
            # += 1
            self.current_frame_float += 1.0
            self.current_frame = int(self.current_frame_float)

        if must_reset:
            done = True

        info = {}
        return self.state, reward, done, info


    def _reset(self):
        # Current Frame
        if self.randomize_start:
            self.current_frame = rd.randint(0, self.frame_count-1-1)
            self.current_frame_float = float(self.current_frame)
        else:
            self.current_frame = 0
            self.current_frame_float = 0.0
        # Current finger position
        current_key = int((self.key_count*0.5)-(self.fingers_count*0.5))
        #for note_pitch, note_val in enumerate(self.frames[self.current_frame]):
        #    if note_val == 1.0 and current_key > note_pitch:
        #        current_key = note_pitch
        self.current_finger_position = []
        for n_finger in range(0, self.fingers_count):
            self.current_finger_position.append(current_key+n_finger)
        # Misc
        self.action_history = []
        self.failed_frames = 0
        self.state = np.array([0 for _ in range((self.key_count * 2) + (self.key_count * self.fingers_count))])
        self.state = self.state.reshape((1, 12, 88, 1), order='F')
        self.state = self.state.transpose(axes=[0, 2, 1, 3])
        self.steps_beyond_done = None
        return np.array(self.state)


    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400
        #
        #world_width = self.x_threshold*2
        #scale = screen_width/world_width
        #cartwidth = 50.0
        #cartheight = 30.0
        fingerwidth = 6.0
        keywidth = 5.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            r_fingers = []
            self.fingertrans = []
            for _ in range(0, self.fingers_count):
                r_fingers.append(rendering.make_circle(fingerwidth/2))
                if _ < 5:
                    col = ((1.*_/5)*.5)+.5
                    r_fingers[-1].set_color(col,0.,0.)
                else:
                    col = (1.-(((1.*(_-5)/5)*.5)+.5))+.5
                    r_fingers[-1].set_color(.0,col,0.)
                self.fingertrans.append(rendering.Transform())
                r_fingers[-1].add_attr(self.fingertrans[-1])
                self.viewer.add_geom(r_fingers[-1])

            r_keys = []
            self.keytrans = []
            for _ in range(0, self.key_count):
                keywidth = 5.0
                keyheight = 15.0
                l,r,t,b = -keywidth/2, keywidth/2, keyheight/2, -keyheight/2
                r_keys.append(rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]))
                r_keys[-1].set_color(.9,.0,.9)
                self.keytrans.append(rendering.Transform())
                r_keys[-1].add_attr(self.keytrans[-1])
                self.viewer.add_geom(r_keys[-1])

            r_keys = []
            self.keypasttrans = []
            for _ in range(0, self.key_count):
                keywidth = 5.0
                keyheight = 15.0
                l,r,t,b = -keywidth/2, keywidth/2, keyheight/2, -keyheight/2
                r_keys.append(rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]))
                r_keys[-1].set_color(.9,.6,.9)
                self.keypasttrans.append(rendering.Transform())
                r_keys[-1].add_attr(self.keypasttrans[-1])
                self.viewer.add_geom(r_keys[-1])


        if self.state is None: return None

        # Transform fingers
        for idx, ft in enumerate(self.fingertrans):
            start = idx*self.key_count
            end = start+self.key_count
            pos = 0
            for idxx in range(start, end):
                if self.state[idxx] == 1.0:
                    pos = idxx-start
            #print (pos)
            self.fingertrans[idx].set_translation(((screen_width/90)*pos)+10, 200)

        # Transform keys
        start = self.key_count*self.fingers_count
        end = start+self.key_count
        for idx, kt in enumerate(self.state[start:end]):
            self.keytrans[idx].set_translation(((screen_width/90)*idx)+10, 180-(kt*5))

        # Transform future keys
        start = (self.key_count*self.fingers_count)+self.key_count
        end = start+self.key_count
        for idx, kt in enumerate(self.state[start:end]):
            self.keypasttrans[idx].set_translation(((screen_width/90)*idx)+10, 140-(kt*5))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
