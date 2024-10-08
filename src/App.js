import React, { useState, useEffect , useRef} from "react";
import axios from "axios";
import Login from "./login";
import TrackApi from "./trackApi";
import CsvDownload from "./CsvDownload";
import '../src/index.css'
import { FaSearch } from "react-icons/fa";
import SearchBar from "./SearchBar";
import MusicPlayer from './MusicPlayer'
// import jwt from 'jsonwebtoken';


function App() {
  const CLIENT_ID = "0461e872c18c40278b9303528b68fbc5";
  const REDIRECT_URI = "http://localhost:3000";
  const AUTH_ENDPOINT = "https://accounts.spotify.com/authorize";
  const RESPONSE_TYPE = "token";
  const SCOPE = "user-top-read user-read-recently-played user-read-playback-state user-modify-playback-state user-read-currently-playing streaming user-read-private playlist-read-collaborative playlist-modify-public playlist-read-private playlist-modify-private user-library-modify user-library-read user-read-playback-position user-read-birthdate user-read-email";
  const playerRef = useRef(null);

  const [token, setToken] = useState("");
  const [trackIds, setTrackIds] = useState([]);
  const [trackFeatures, setTrackFeatures] = useState([]);
  const [loggedIn, setLoggedIn] = useState(true);
  const [trackDetails, setTrackDetails] = useState([]);
  const [getIds,setGetIds] = useState([])
  const [songId, setSongId] = useState('2takcwOaAZWiXQijPHIx7B'); // Example song ID
  const [trackURIs, setTrackURIs] = useState([]);

//   const jwt = require('jsonwebtoken');

// const accessToken = localStorage.getItem('token'); // Get your access token from local storage

// if (accessToken) {
//   const decodedToken = jwt.decode(accessToken);
//   if (decodedToken) {
//     console.log('Decoded Token:', decodedToken);
//     console.log('Scopes:', decodedToken.scope); // Access the scopes property
//   } else {
//     console.error('Failed to decode access token');
//   }
// } else {
//   console.error('Access token not found in local storage');
// }


  useEffect(() => {
    const hash = window.location.hash;
    let token = window.localStorage.getItem("token");

    if (!token && hash) {
      token = hash.substring(1).split("&").find(elem => elem.startsWith("access_token")).split("=")[1];
      window.location.hash = "";
      window.localStorage.setItem("token", token);
    }

    setToken(token);
  }, []);

  useEffect(() => {
    if (token) {
      fetchTopTracks();
    }
  }, [token]);

  useEffect(() => {
    if (getIds.length > 0) {
      fetchTrackDetailsForAll(getIds);
    }
  }, [getIds]);

  const fetchTopTracks = async () => {
    try {
      const trackApi = new TrackApi(token);
      const trackIdsArray = await trackApi.getTopTrackIds();
      setTrackIds(trackIdsArray);
      const trackFeaturesArray = await trackApi.getTrackFeatures(trackIdsArray);
      setTrackFeatures(trackFeaturesArray);
    } catch (error) {
      console.error('Error fetching top tracks:', error);
    }
  };

  const fetchTrackDetailsForAll = async (trackIds) => {
    try {
      const trackDetailsArray = [];
      for (const trackId of trackIds) {
        const trackDetails = await fetchTrackDetails(trackId);
        trackDetailsArray.push(trackDetails);
      }
      setTrackDetails(trackDetailsArray);
    } catch (error) {
      console.error('Error fetching track details for all IDs:', error);
    }
  };

  const fetchTrackDetails = async (trackId) => {
    try {
        const response = await axios.get(`https://api.spotify.com/v1/tracks/${trackId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        console.log("Track details for track ID", trackId, ":", response.data);

        // Extract the URI from the response and store it in an array
        const uri = response.data.uri;
        setTrackURIs(prevTrackURIs => [...prevTrackURIs, uri]);

        return uri;
    } catch (error) {
        console.error('Error fetching track details:', error);
        return null;
    }
};


console.log("this is are traclui",trackURIs)

  const handleLogout = () => {
    setLoggedIn(false);
    setToken("");
    window.localStorage.removeItem("token");
  };

  const handleGenerateRecommendations = async () => {
    await sendFileToBackend();
    // You can add additional code here if needed
  };
  useEffect(() => {
    if (playerRef.current) {
      playerRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [trackURIs.length==5]);
  const sendFileToBackend = async () => {
    const fileData = localStorage.getItem('music_features_csv');
    console.log("fileData", fileData);
    if (fileData) {
      try {
        const response = await axios.post('http://127.0.0.1:8000/process-csv/', fileData, {
          headers: {
            'Content-Type': 'text/csv'
          }
        });
        const { track_ids } = response.data;
        setGetIds(track_ids);
        console.log('Track IDs received:', track_ids);
        console.log('File sent to backend successfully');
      } catch (error) {
        console.error('Error sending file to backend:', error);
      }
    } else {
      console.error('File not found in local storage');
    }
  };
  
  return (
    <div>
      <div className="flex flex-col min-h-screen gradient-animation">
        <div className="container mx-auto flex justify-between items-center px-4 py-8">
          {/* Render logout and generate recommendations buttons based on token */}
        </div>
        <div className="flex justify-center items-center flex-grow h-10">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-white text-7xl font-nunito">Music Recommendation System</h1>
            <p className="text-white text-xl mt-16">Log in to your Spotify account and we’ll serve you up with 5 similar tracks that you’ll love!</p>
            <div className=" ">
              {!token ? <Login /> : (
                <div className="text-center justify-center gap-2 mt-8">
                  <div className="flex text-center justify-center gap-2">
                    <SearchBar/>
                    <FaSearch className=" flex text-white text-2xl justify-center"/>
                    </div>
                    <div className="flex text-center justify-center items-center gap-2 mt-8"> 
                  <button onClick={handleGenerateRecommendations} className="bg-transparent hover:bg-white hover:text-black border-2 border-white px-8 py-4 rounded-lg text-white text-lg font-bold transition duration-300 ease-in-out mr-4">
                    Generate Recommendations
                  </button>
                  {loggedIn && (
                    <button onClick={handleLogout} className="bg-transparent hover:bg-white hover:text-black border-2 border-white px-8 py-4 rounded-lg text-white text-lg font-bold transition duration-300 ease-in-out">
                      Logout
                    </button>
                  )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="container mx-auto ">
          {/* Your existing JSX for recommendations */}
          {trackFeatures.length > 0 && <CsvDownload data={trackFeatures} />}
        </div>
      </div>
      <div ref={playerRef}>
         {token && trackURIs.length >= 5 && (
        <div className="flex justify-center items-center h-full">
 <MusicPlayer accessToken={token} trackUri={trackURIs} />        </div>
      )}
        
      </div>
    </div>
  );
}

export default App;
