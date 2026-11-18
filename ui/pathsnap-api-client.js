/* PathSnap API adapter; include with <script src="pathsnap-api-client.js"></script> */
(function(){
  const normalize = base => String(base || '').replace(/\/+$/,'');
  async function readJson(url, options={}){
    const response=await fetch(url,options);
    let data;
    try{data=await response.json()}catch(e){throw Error('Non-JSON response: HTTP '+response.status)}
    if(!response.ok)throw Error(data.detail || 'HTTP '+response.status);
    return data;
  }
  window.PathSnapAPI={
    listTrips(base){return readJson(normalize(base)+'/trips').then(x=>x.trips);},
    async predict(base,tripId){
      const id=String(tripId).padStart(8,'0');
      if(!/^\d{8}$/.test(id))throw Error('Invalid trip id');
      return readJson(normalize(base)+'/predict',{
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({trip_id:id})
      });
    }
  };
})();
