import React, { useState, useEffect } from "react";
import { FaPlay ,FaPause} from "react-icons/fa";
import { MdOutlinePlayArrow, MdSkipNext, MdSkipPrevious } from "react-icons/md";

const TrackPlayer = ({ trackDetails }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [currentTrackIndex, setCurrentTrackIndex] = useState(0);

  useEffect(() => {
    if (trackDetails.length > 0) {
      setDuration(trackDetails[currentTrackIndex].duration_ms / 1000);
    }
  }, [trackDetails, currentTrackIndex]);

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleTimeUpdate = (e) => {
    setCurrentTime(e.target.currentTime);
  };

  const handleDurationChange = (e) => {
    setDuration(e.target.duration);
  };

  const handleSeek = (e) => {
    const seekTime = e.target.value;
    setCurrentTime(seekTime);
  };

  const handleNextTrack = () => {
    setCurrentTrackIndex((prevIndex) => (prevIndex + 1) % trackDetails.length);
    setCurrentTime(0);
  };

  const handlePreviousTrack = () => {
    setCurrentTrackIndex((prevIndex) => (prevIndex === 0 ? trackDetails.length - 1 : prevIndex - 1));
    setCurrentTime(0);
  };

  return (
    <div className="flex flex-col items-center justify-center w-96 pb-8 bg-transparent rounded-lg shadow-lg mb-28">
      <div className="w-48 h-48 rounded-full bg-white mb-4"></div>

      <div className="flex items-center mb-4 text-white">
        <MdSkipPrevious className="text-white text-3xl mr-4 cursor-pointer" onClick={handlePreviousTrack} />
        {isPlaying ? (
          <FaPause className="text-white text-3xl mr-4 cursor-pointer" onClick={togglePlayPause} />
        ) : (
          <FaPlay className="text-white text-3xl mr-4 cursor-pointer" onClick={togglePlayPause} />
        )}
        <MdSkipNext className="text-white text-3xl cursor-pointer" onClick={handleNextTrack} />
      </div>

      <div className="progress flex flex-col items-center w-full">
        <span className="time current text-white text-lg mb-2">{formatTime(currentTime)}</span>
        <input
          type="range"
          className="progress-bar w-full"
          onChange={handleSeek}
          value={currentTime}
          max={duration}
          style={{
            background: 'transparent',
            height: '6px',
            borderRadius: '3px',
            outline: 'none',
            cursor: 'pointer',
          }}
        />
        <span className="time text-white text-lg mt-2">{formatTime(duration)}</span>
      </div>
    </div>
  );
};

// Function to format time in mm:ss format
const formatTime = (time) => {
  const minutes = Math.floor(time / 60);
  const seconds = Math.floor(time % 60);
  return `${minutes}:${seconds < 10 ? "0" : ""}${seconds}`;
};

export default TrackPlayer;
