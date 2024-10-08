// src/components/TrackApi.js
import axios from 'axios';

class TrackApi {
  constructor(token) {
    this.token = token;
  }

  async getTopTrackIds() {
    try {
      const response = await axios.get('https://api.spotify.com/v1/me/top/tracks?time_range=medium_term&limit=50', {
        headers: {
          'Authorization': `Bearer ${this.token}`
        }
      });
      console.log("this is top track", response.data.items)
  
      return response.data.items.map(item=>item.id);
      
    } catch (error) {
      throw new Error('Error fetching top tracks:', error);
    }
  }
  async getRecentlyPlayed() {
    try {
      const response = await axios.get('https://api.spotify.com/v1/me/player/recently-played', {
        headers: {
          'Authorization': `Bearer ${this.token}`
        }
      });
      console.log("this is recently played songs",response.data.items)
  
      return response.data.items.map(item=>item.id);
      
    } catch (error) {
      throw new Error('Error fetching top tracks:', error);
    } 
  }

  async getTrackFeatures(trackIds) {
    try {
      const promises = trackIds.map(async (id) => {
        const response = await axios.get(`https://api.spotify.com/v1/audio-features/${id}`, {
          headers: {
            'Authorization': `Bearer ${this.token}`
          }
        });
        console.log("this is response data", response.data)
        return response.data;
      });

      return Promise.all(promises);
    } catch (error) {
      throw new Error('Error fetching track features:', error);
    }
  }
}



export default TrackApi;
