const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
let threshold=.63, maxFiles=20, files=[], activeResult=null, filter='all';
let currentAnalysisStep=1;
let claimPage=1,claimSortKey='risk',claimSortDirection='desc';
const claimsPerPage=10;
let saveTimer=null, requestId=null, bridgeReady=false, analysisResults=[], previewUrls=[], pendingClaimInfo=null;
let claims=[], selectedClaimId=null;
const claimsReady=claimStore.list().then(async saved=>{
  const legacyAutoCompleted=saved.filter(claim=>claim.status==='심사완료'&&!claim.reviewCompletedAt);
  claims=saved.map(claim=>legacyAutoCompleted.includes(claim)?{...claim,status:'심사대기'}:claim);
  await Promise.all(claims.filter(claim=>legacyAutoCompleted.some(old=>old.id===claim.id)).map(claim=>claimStore.put(claim).catch(()=>{})));
  renderClaims();return true;
}).catch(()=>{toast('저장된 청구건을 불러오지 못했습니다. 브라우저 저장소 사용 설정을 확인해 주세요.');return false});
let roiCandidate=null;
let roiManualAverageMode=false;
let storedRoiSettings=null,roiMetaLoaded=false,roiThresholdSynced=false;
claimStore.getMeta('roi-settings').then(settings=>{storedRoiSettings=settings||null;roiMetaLoaded=true}).catch(()=>{roiMetaLoaded=true});
const escapeHtml=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const modelNames={trufor:'TruFor',fused:'FUSED',recapture:'Recapture'};
function generateClaimId(){
  const year=new Date().getFullYear();
  let suffix;
  do{const random=globalThis.crypto?.getRandomValues?crypto.getRandomValues(new Uint32Array(1))[0]:Math.floor(Math.random()*1e9);suffix=String((Date.now()+random)%1000000).padStart(6,'0')}while(claims.some(claim=>claim.id===`MLM-${year}-${suffix}`));
  return `MLM-${year}-${suffix}`;
}
function generateClaimAmount(){return Math.round((500000+Math.floor(Math.random()*(2000000-500000+1)))/100)*100}
function customerFormData(){return {customerName:$('#customerName').value.trim(),customerBirth:$('#customerBirth').value.trim(),customerPhone:$('#customerPhone').value.trim(),accidentDate:$('#accidentDate').value,accidentCause:$('#accidentCause').value.trim()}}
function customerRequiredFieldsReady(){return Boolean($('#customerName').value.trim()&&$('#customerPhone').value.trim())}
function updateAnalyzeButton(){$('#analyzeBtn').disabled=!files.length||!bridgeReady||!!requestId||!customerRequiredFieldsReady()}
['#customerName','#customerPhone'].forEach(selector=>$(selector).addEventListener('input',updateAnalyzeButton));
function sendComponent(type,payload={}){window.parent.postMessage({isStreamlitMessage:true,type,...payload},'*')}
function sendEvent(event){sendComponent('streamlit:setComponentValue',{value:event,dataType:'json'})}
function eventId(){return globalThis.crypto?.randomUUID?.()||`${Date.now()}-${Math.random()}`}
const tone=v=>v>=.7?'high':v>=.4?'medium':'low';
function setProcessStage(stage){$$('.nav-btn[data-process]').forEach(button=>{const value=Number(button.dataset.process);button.classList.toggle('active',value===stage);if(value===stage)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current')})}
function showView(name){
  $$('.view').forEach(v=>v.classList.remove('active'));$(`#${name}View`).classList.add('active');
  setProcessStage(['claims','claimDetail','imageDetail','roiSettings'].includes(name)?3:currentAnalysisStep===1?1:2);
  if(name==='claims')renderClaims();scrollTo({top:0,behavior:'smooth'});
}
$$('.nav-btn[data-process]').forEach(button=>button.onclick=()=>{
  const stage=Number(button.dataset.process);
  if(stage===1){goHome();return}
  if(stage===2){showView('analysis');if(currentAnalysisStep===1)toast('1단계에서 청구 정보와 이미지를 제출하면 내부 이미지 처리가 시작됩니다.');return}
  showView('claims');
});
$('#brandBtn').onclick=goHome;$('#backClaimsBtn').onclick=()=>showView('claims');
$('#roiSettingsBtn').onclick=()=>openRoiSettings();$('#backRoiBtn').onclick=()=>showView('claims');
$('#themeBtn').onclick=()=>{document.body.classList.toggle('light');localStorage.setItem('cg-theme-navy-v1',document.body.classList.contains('light')?'light':'dark')};if(localStorage.getItem('cg-theme-navy-v1')==='light')document.body.classList.add('light');
const dz=$('#dropzone'),fi=$('#fileInput');['dragenter','dragover'].forEach(e=>dz.addEventListener(e,x=>{x.preventDefault();dz.classList.add('drag')}));['dragleave','drop'].forEach(e=>dz.addEventListener(e,x=>{x.preventDefault();dz.classList.remove('drag')}));dz.addEventListener('drop',e=>setFiles(e.dataTransfer.files,{append:true}));fi.onchange=e=>{setFiles(e.target.files,{append:true});fi.value=''};
function setFiles(list,{append=false}={}){
  const selected=[...list],valid=selected.filter(f=>/^image\/(jpeg|png)$/.test(f.type)&&f.size<=20*1024*1024);
  if(valid.length!==selected.length)toast('20MB 이하의 JPEG/PNG 파일만 선택할 수 있습니다.');
  const next=append?[...files,...valid]:valid;
  if(next.length>maxFiles||next.reduce((sum,f)=>sum+f.size,0)>100*1024*1024){toast(`최대 ${maxFiles}장, 전체 100MB까지 선택해 주세요.`);return}
  previewUrls.forEach(url=>URL.revokeObjectURL(url));previewUrls=[];files=next;
  $('#fileSummary').textContent=files.length?`이미지 ${files.length}장이 첨부되었습니다.`:'첨부된 이미지가 없습니다.';
  $('#fileList').innerHTML=files.map((f,index)=>{const url=URL.createObjectURL(f);previewUrls.push(url);return `<div class="file-pill"><img src="${url}" alt=""><span>${escapeHtml(f.name)}</span><small>${(f.size/1048576).toFixed(1)} MB</small><button class="remove-file" type="button" data-remove-file="${index}" aria-label="${escapeHtml(f.name)} 첨부 취소">×</button></div>`}).join('');
  $$('#fileList [data-remove-file]').forEach(button=>button.onclick=()=>removeFile(Number(button.dataset.removeFile)));
  updateAnalyzeButton();
}
function removeFile(index){
  if(!Number.isInteger(index)||index<0||index>=files.length)return;
  setFiles(files.filter((_,fileIndex)=>fileIndex!==index));
}
function filePayload(file){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve({name:file.name,data:reader.result.split(',')[1]});reader.onerror=()=>reject(new Error('이미지를 읽지 못했습니다.'));reader.readAsDataURL(file)})}
$('#analyzeBtn').onclick=async()=>{
  if(!customerRequiredFieldsReady()){toast('이름과 연락처를 입력해 주세요.');return}
  if(!files.length){toast('사고 이미지를 한 장 이상 첨부해 주세요.');return}
  if(!bridgeReady||requestId)return;
  const claimId=generateClaimId(),claimAmount=generateClaimAmount();$('#claimId').value=claimId;$('#claimAmount').value=String(claimAmount);pendingClaimInfo=customerFormData();
  const id=eventId();requestId=id;analysisResults=[];$('#analyzeBtn').disabled=true;
  $('#uploadStage').classList.add('hidden');$('#resultStage').classList.add('hidden');$('#analyzingStage').classList.remove('hidden');
  $('#scanImage').src=previewUrls[0];$('#analysisStatus').textContent='이미지를 분석 서버로 전달하고 있습니다.';setStep(2);
  try{const payload=await Promise.all(files.map(filePayload));if(requestId===id)sendEvent({id,action:'analyze',claimId,...pendingClaimInfo,files:payload,threshold})}
  catch(error){if(requestId===id){resetAnalysis();toast(error.message)}}
};
function receiveJob(job){
  if(!job||job.id!==requestId)return;
  if(job.status==='error'){const message=job.error;resetAnalysis();toast(message);return}
  if(job.status==='cancelled'){resetAnalysis();return}
  if(job.status==='complete'){
    if(analysisResults.length)return;
    analysisResults=job.results;$('#saveBtn').disabled=false;
    $('#resultPicker').innerHTML=analysisResults.map((r,i)=>`<option value="${i}">${escapeHtml(r.name)}${r.complete?'':' · 분석 미완료'}</option>`).join('');
    selectResult(0);return;
  }
  $('#analysisStatus').textContent=job.status==='queued'?'분석 준비 중입니다. 다른 분석이 실행 중이면 완료 후 시작합니다.':`${job.image_index+1}/${job.total} · ${job.name} · ${modelNames[job.model]||'모델'} 실행 중`;
  if(previewUrls[job.image_index])$('#scanImage').src=previewUrls[job.image_index];
  $$('.model-task').forEach((element,i)=>{const name=Object.keys(modelNames)[i],result=job.models[name];element.classList.toggle('active',name===job.model);element.classList.toggle('failed',result?.status==='error');element.querySelector('small').textContent=result?(result.status==='complete'?'완료':'실행 실패'):name===job.model?'분석 중':'대기 중'})
}
window.addEventListener('message',event=>{
  if(event.source!==window.parent||event.data?.type!=='streamlit:render')return;
  const args=event.data.args||{};bridgeReady=true;threshold=args.threshold??.63;
  if(Number.isInteger(args.max_files)&&args.max_files>0)maxFiles=args.max_files;
  $('#uploadRules').textContent=`JPEG 또는 PNG · 이미지당 20MB · 최대 ${maxFiles}장 / 전체 100MB`;
  updateAnalyzeButton();
  $('.system-state').innerHTML='<i></i>분석 서버 연결됨';
  updateThresholdUi();
  receiveJob(args.job);
  if(roiMetaLoaded&&!roiThresholdSynced){
    roiThresholdSynced=true;
    if(storedRoiSettings&&Number.isFinite(storedRoiSettings.threshold)&&Math.abs(storedRoiSettings.threshold-threshold)>.000001){
      threshold=storedRoiSettings.threshold;updateThresholdUi();
      sendEvent({id:eventId(),action:'set_roi_threshold',threshold});
    }
  }
});
function updateThresholdUi(){
  $$('.threshold-card strong,#resultStage .threshold-line b').forEach(el=>el.textContent=Math.round(threshold*100)+'%');
  $('#thresholdMark').style.left=(threshold*100)+'%';
  $('#roiCurrentThreshold').textContent=percent(threshold);
}
function selectResult(index){
  if(!analysisResults.length)return;
  const selectedIndex=Math.max(0,Math.min(Number(index)||0,analysisResults.length-1));
  activeResult=analysisResults[selectedIndex];$('#resultPicker').value=String(selectedIndex);
  $('#resultPosition').textContent=`${selectedIndex+1} / ${analysisResults.length}`;
  $('#previousResultBtn').disabled=selectedIndex===0;
  $('#nextResultBtn').disabled=selectedIndex===analysisResults.length-1;
  $('#resultNavigation').classList.toggle('hidden',analysisResults.length<2);
  showResult();
}
function setStep(n){
  currentAnalysisStep=n;setProcessStage(n===1?1:2);
  $('#analysisHeading').textContent={1:'차량 손상 이미지 등록',2:'이미지 포렌식',3:'결과 검토',4:'저장'}[n]||'차량 손상 이미지 등록';
  $$('.step').forEach(s=>{const v=+s.dataset.step;s.classList.toggle('active',v===n);s.classList.toggle('done',v<n);if(v===n)s.setAttribute('aria-current','step');else s.removeAttribute('aria-current')});
  const progress=((n-1)/3*100)+'%';
  $('#journeyCar').style.left=progress;
  $('#journeyFill').style.width=progress;
}
function showResult(){
  setStep(3);$('#analyzingStage').classList.add('hidden');$('#resultStage').classList.remove('hidden');
  $('#generatedClaimId').textContent=$('#claimId').value||'—';
  $('#generatedClaimAmount').textContent=Number($('#claimAmount').value)>0?won(Number($('#claimAmount').value)):'—';
  const riskScore=resultRiskScore(activeResult),valid=isComplete(activeResult)&&Number.isFinite(riskScore),review=valid&&riskScore>=resultThreshold(activeResult),pct=valid?Math.round(riskScore*100):null;
  $('#finalScore').textContent=$('#fusionValue').textContent=valid?pct+'%':'—';
  $('#riskRing').style.setProperty('--score',(pct??0)+'%');$('#scoreMark').style.setProperty('--left',(pct??0)+'%');$('#scoreMark').hidden=!valid;
  $('#verdictCard').classList.toggle('safe',valid&&!review);$('#resultStage').dataset.risk=valid?tone(riskScore):'medium';
  $('#verdictTitle').textContent=!valid?'모델 분석을 완료하지 못했습니다':review?'이미지 조작 가능성이 높습니다':'정상 범위 입니다';
  $('#verdictText').textContent=!valid?'실패한 모델을 확인해 주세요. 세 모델이 모두 완료되어야 최종 위험도를 계산합니다.':review?'세 모델의 통합 신호가 ROI 임계값 이상입니다.':'세 모델의 통합 신호가 현재 ROI 임계값보다 낮습니다.';
  $('#routeResult').textContent=!valid?'판정 보류 · 모델 실행 오류 확인':review?'임계값 초과':'임계값 범위';
  $('#modelScores').innerHTML=modelScoresMarkup(activeResult);
  const defaultOverlay=activeResult.artifacts.heatmap?'heatmap':'original';
  $$('[data-overlay]').forEach(b=>{b.disabled=b.dataset.overlay!=='original'&&!activeResult.artifacts[b.dataset.overlay];b.classList.toggle('active',b.dataset.overlay===defaultOverlay)});
  $('#overlayNote').textContent=activeResult.artifacts.heatmap?'FUSED 실제 추론 결과입니다. 표시 영역은 검토 보조 신호이며 보험사기 확정 판정이 아닙니다.':'FUSED 위치 분석 결과가 없어 원본만 표시합니다. 모델별 실행 상태를 확인해 주세요.';
  drawOverlay(defaultOverlay);
}
function drawOverlay(mode){$('#resultImage').src=mode==='original'?activeResult.url:activeResult.artifacts[mode]||activeResult.url}
$$('[data-overlay]').forEach(b=>b.onclick=()=>{$$('[data-overlay]').forEach(x=>x.classList.remove('active'));b.classList.add('active');drawOverlay(b.dataset.overlay)});
$('#newAnalysisBtn').onclick=()=>resetAnalysis();
function goHome(){resetAnalysis();showView('analysis')}
function resetAnalysis(){
  if(requestId)sendEvent({id:eventId(),action:'cancel'});
  requestId=null;clearTimeout(saveTimer);saveTimer=null;activeResult=null;analysisResults=[];pendingClaimInfo=null;$('#claimId').value='';
  $('#analyzingStage').classList.add('hidden');$('#resultStage').classList.add('hidden');$('#uploadStage').classList.remove('hidden');
  $$('.model-task').forEach(e=>{e.classList.remove('active','failed');e.querySelector('small').textContent='대기 중'});
  $('#saveBtn').disabled=false;fi.value='';setFiles([]);setStep(1);
}
$('#resultPicker').onchange=e=>selectResult(Number(e.target.value));
$('#previousResultBtn').onclick=()=>selectResult(Number($('#resultPicker').value)-1);
$('#nextResultBtn').onclick=()=>selectResult(Number($('#resultPicker').value)+1);
$('#downloadResultsBtn').onclick=()=>{
  const data=analysisResults.map(({url,artifacts,...result})=>result);
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  const anchor=document.createElement('a');anchor.href=url;anchor.download='claimguard-model-results.json';anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
function resultThreshold(result){return Number.isFinite(result.threshold)?result.threshold:.63}
function isComplete(result){return result.complete===true&&Number.isFinite(result.score)}
function resultRiskScore(result){return highestModelSignal(result)?.score??(Number.isFinite(result.score)?result.score:null)}
function needsReview(result){const score=resultRiskScore(result);return isComplete(result)&&Number.isFinite(score)&&score>=resultThreshold(result)}
function claimCategory(claim){return !claim.images.length?'empty':claim.images.some(r=>!isComplete(r))?'incomplete':claim.images.some(needsReview)?'review':'normal'}
function categoryLabel(category){return {review:'검토대상',normal:'정상추정',incomplete:'분석미완료',empty:'사진 없음'}[category]}
function inspectionLabel(value){return value==='approved'?'승인':value==='returned'?'반송':'미결정'}
function percent(value){return Number.isFinite(value)?(value*100).toFixed(1)+'%':'—'}
function won(value){return `${Math.round(value).toLocaleString()}원`}
function modelScoresMarkup(result){
  return Object.entries(modelNames).map(([key,name])=>{
    const model=result.models?.[key],score=model?.score??result.scores?.[key],ok=Number.isFinite(score)&&model?.status!=='error';
    const elapsed=Number.isFinite(model?.elapsed_sec)?` · ${model.elapsed_sec.toFixed(1)}초`:'';
    return `<div class="model-score ${ok?tone(score):'failed'}"><header><div><b>${name}</b><small>${ok?'모델 분석 완료'+elapsed:'분석 미완료'}</small></div><span>${percent(ok?score:null)}</span></header>${ok?`<div class="bar"><i style="width:${score*100}%"></i></div>`:`<p class="model-error">${escapeHtml(model?.error||'모델 점수가 없습니다.')}</p>`}</div>`;
  }).join('');
}
function highestModelSignal(result){
  return Object.entries(modelNames).map(([key,name])=>{
    const model=result.models?.[key],score=model?.score??result.scores?.[key];
    return Number.isFinite(score)&&model?.status!=='error'?{key,name,score}:null;
  }).filter(Boolean).reduce((highest,item)=>!highest||item.score>highest.score?item:highest,null);
}
$('#saveBtn').onclick=async()=>{
  if(!analysisResults.length||$('#saveBtn').disabled)return;
  $('#saveBtn').disabled=true;
  if(!await claimsReady){$('#saveBtn').disabled=false;toast('저장소를 사용할 수 없습니다. 결과 JSON을 내려받아 주세요.');return}
  // Capture the current batch before an asynchronous write or navigation.
  const batch=analysisResults.map(({artifacts,...result})=>({...result,threshold:Number.isFinite(result.threshold)?result.threshold:threshold,artifacts:artifacts?.heatmap?{heatmap:artifacts.heatmap}:{}}));
  const savingRequest=requestId;
  const id=$('#claimId').value||generateClaimId();$('#claimId').value=id;
  const claimInfo=pendingClaimInfo||customerFormData();
  const existing=claims.find(c=>c.id===id);
  const amount=+$('#claimAmount').value||generateClaimAmount();$('#claimAmount').value=String(amount);
  const claim=existing?{...existing,images:[...existing.images],amount:existing.amount||amount}:{id,date:new Date().toISOString().slice(0,10),amount,images:[]};
  Object.assign(claim,claimInfo);
  claim.images.push(...batch);
  claim.status='심사대기';
  delete claim.reviewCompletedAt;
  try{await claimStore.put(claim)}catch(error){$('#saveBtn').disabled=false;toast('사진과 분석 결과를 저장하지 못했습니다. 브라우저 저장 공간을 확인해 주세요.');return}
  claims=[claim,...claims.filter(c=>c.id!==id)];renderClaims();
  toast(`${batch.length}장의 사진과 분석 결과를 저장했습니다.`);
  if(requestId!==savingRequest)return;
  setStep(4);saveTimer=setTimeout(()=>{saveTimer=null;showView('claims')},1000);
};
function maxScore(c){return Math.max(0,...c.images.map(resultRiskScore).filter(Number.isFinite))}
function renderClaims(){
  const review=claims.filter(c=>claimCategory(c)==='review').length,normal=claims.filter(c=>claimCategory(c)==='normal').length;
  $('#claimCount').textContent=claims.length;
  $('#kpis').innerHTML=`<div class="kpi"><span>전체 청구</span><strong>${claims.length}</strong></div><div class="kpi red"><span>검토대상</span><strong>${review}</strong></div><div class="kpi green"><span>정상추정</span><strong>${normal}</strong></div>`;
  renderRows();
}
function renderRows(){
  const query=$('#searchInput').value.toLowerCase();
  const list=claims.filter(c=>(filter==='all'||filter===claimCategory(c))&&(c.id.toLowerCase().includes(query)||c.images.some(i=>i.name.toLowerCase().includes(query))));
  const direction=claimSortDirection==='asc'?1:-1;
  list.sort((a,b)=>{
    if(claimSortKey==='amount')return (Number(a.amount||0)-Number(b.amount||0))*direction;
    if(claimSortKey==='date')return String(a.date||'').localeCompare(String(b.date||''))*direction;
    return (maxScore(a)-maxScore(b))*direction;
  });
  const totalPages=Math.max(1,Math.ceil(list.length/claimsPerPage));
  claimPage=Math.min(Math.max(1,claimPage),totalPages);
  const pageItems=list.slice((claimPage-1)*claimsPerPage,claimPage*claimsPerPage);
  $('#claimRows').innerHTML=pageItems.map(c=>{
    const category=claimCategory(c),color=category==='review'?'high':category==='normal'?'low':'medium';
    const inspection=c.inspectionResult||'pending';
    return `<div class="claim-row" data-id="${escapeHtml(c.id)}"><span class="claim-id"><button class="claim-open" data-open-claim="${escapeHtml(c.id)}">${escapeHtml(c.id)}</button></span><span class="claim-amount">${Number(c.amount||0).toLocaleString()}원</span><span>${c.images.length}장</span><span>${escapeHtml(c.date)}</span><span class="risk ${color}">${category==='empty'?'—':category==='incomplete'?'미완료':percent(maxScore(c))}</span><span class="analysis-signal ${category}">${categoryLabel(category)}</span><span class="inspection-result ${inspection}">${inspectionLabel(inspection)}</span></div>`;
  }).join('')||'<div class="empty-state">조건에 맞는 청구가 없습니다.</div>';
  renderClaimPagination(list.length,totalPages);
  $$('.claim-row[data-id]').forEach(row=>row.onclick=()=>openClaim(row.dataset.id));
}
function renderClaimPagination(total,totalPages){
  $('#claimPagination').innerHTML=`<span>총 ${total.toLocaleString()}건 · ${claimPage}/${totalPages}페이지</span><div><button class="secondary compact" data-page="prev" ${claimPage===1?'disabled':''}>이전</button><button class="secondary compact" data-page="next" ${claimPage===totalPages?'disabled':''}>다음</button></div>`;
  $$('#claimPagination [data-page]').forEach(button=>button.onclick=()=>{claimPage+=button.dataset.page==='next'?1:-1;renderRows();$('#claimsView').scrollIntoView({behavior:'smooth'})});
}
$$('#filters button').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;claimPage=1;$$('#filters button').forEach(x=>x.classList.toggle('active',x===b));renderRows()});
$('#searchInput').oninput=()=>{claimPage=1;renderRows()};
$$('.column-sort').forEach(button=>button.onclick=()=>{
  const key=button.dataset.sort;
  if(claimSortKey===key)claimSortDirection=claimSortDirection==='asc'?'desc':'asc';
  else{claimSortKey=key;claimSortDirection=key==='amount'?'asc':'desc'}
  claimPage=1;
  $$('.column-sort').forEach(item=>{const active=item.dataset.sort===claimSortKey;item.classList.toggle('active',active);item.querySelector('i').textContent=active?(claimSortDirection==='asc'?'↑':'↓'):'↕'});
  renderRows();
});
function imageMarkup(source,alt){
  return typeof source==='string'&&/^data:image\/(jpeg|png);base64,/.test(source)?`<img src="${escapeHtml(source)}" alt="${escapeHtml(alt)}" loading="lazy">`:'<div class="empty-state">저장된 이미지가 없습니다.</div>';
}
function claimInformationMarkup(claim){
  const fields=[['이름',claim.customerName],['생년월일',claim.customerBirth],['연락처',claim.customerPhone],['사고일자',claim.accidentDate]];
  return `<section class="claim-information" aria-label="고객 및 사고 정보"><div class="claim-information-grid">${fields.map(([label,value])=>`<div><span>${label}</span><b>${escapeHtml(value||'—')}</b></div>`).join('')}</div><div class="accident-cause"><span>사고원인 및 경위</span><p>${escapeHtml(claim.accidentCause||'작성된 사고 경위가 없습니다.')}</p></div></section>`;
}
async function deleteClaim(id,button){
  if(!confirm(`${id} 청구건과 저장된 이미지 및 분석 결과를 모두 삭제할까요?`))return;
  button.disabled=true;
  try{await claimStore.remove(id)}catch(error){button.disabled=false;toast('청구건을 삭제하지 못했습니다.');return}
  claims=claims.filter(item=>item.id!==id);selectedClaimId=null;renderClaims();showView('claims');toast('청구건을 삭제했습니다.');
}
function closeInspectionCompletion(){
  const modal=$('#inspectionCompletionModal');
  modal.hidden=true;document.body.classList.remove('modal-open');
}
function showInspectionCompletion(claim,value){
  const approved=value==='approved',modal=$('#inspectionCompletionModal');
  modal.dataset.inspection=value;
  $('#inspectionCompletionIcon').textContent=approved?'✓':'!';
  $('#inspectionCompletionTitle').textContent=approved?'다음 심사단계 진행':'추가 반송절차 진행';
  modal.hidden=false;document.body.classList.add('modal-open');$('#completionBackBtn').focus();
}
$('#completionBackBtn').onclick=()=>{closeInspectionCompletion();showView('claims')};
async function setInspectionResult(id,value,button){
  if(button.disabled)return;
  const claim=claims.find(item=>item.id===id);if(!claim)return;
  $$('#inspectionControls button').forEach(item=>item.disabled=true);
  const updated={...claim,inspectionResult:value,inspectionUpdatedAt:new Date().toISOString()};
  try{await claimStore.put(updated)}catch(error){$$('#inspectionControls button').forEach(item=>item.disabled=false);toast('점검결과를 저장하지 못했습니다.');return}
  claims=claims.map(item=>item.id===id?updated:item);renderClaims();openClaim(id);showInspectionCompletion(updated,value);
}
function openClaim(id){
  const claim=claims.find(c=>c.id===id);if(!claim)return;
  selectedClaimId=id;
  const inspection=claim.inspectionResult||'pending';
  $('#claimDetail').innerHTML=`<div class="detail-head"><div><p class="eyebrow">CLAIM DETAIL</p><div class="claim-title-row"><h1>${escapeHtml(id)}</h1><div class="inspection-controls" id="inspectionControls" aria-label="점검결과 선택"><button class="inspection-choice approved ${inspection==='approved'?'selected':''}" data-inspection="approved">승인</button><button class="inspection-choice returned ${inspection==='returned'?'selected':''}" data-inspection="returned">반송</button></div></div><p>${claim.images.length}장 · ${escapeHtml(claim.date)} 접수 · 청구금액 ${won(Number(claim.amount)||0)}</p></div><div class="claim-detail-actions"><span class="analysis-signal ${claimCategory(claim)}">${categoryLabel(claimCategory(claim))}</span><span class="inspection-result ${inspection}">${inspectionLabel(inspection)}</span><button class="delete-claim-btn" id="deleteDetailClaimBtn" type="button">청구건 삭제</button></div></div>${claimInformationMarkup(claim)}<div class="photo-grid">${claim.images.map((result,index)=>{const riskScore=resultRiskScore(result),signal=!isComplete(result)?'incomplete':needsReview(result)?'review':'normal';return `<article class="photo-item"><button class="photo-card" data-index="${index}" aria-label="${escapeHtml(result.name)} 분석 상세 보기">${imageMarkup(result.url,result.name)}<div class="body"><span class="analysis-signal ${signal}">${categoryLabel(signal)}</span><h3>${escapeHtml(result.name)}</h3><div class="meta"><span>분석 상세 보기 →</span><b class="risk ${isComplete(result)?tone(riskScore):'medium'}">${percent(isComplete(result)?riskScore:null)}</b></div></div></button></article>`}).join('')||'<div class="empty-state">등록된 사진이 없습니다. 이미지 분석 후 저장하면 사진을 추가할 수 있습니다.</div>'}</div>`;
  $$('#inspectionControls [data-inspection]').forEach(button=>button.onclick=()=>setInspectionResult(id,button.dataset.inspection,button));
  $('#deleteDetailClaimBtn').onclick=event=>deleteClaim(id,event.currentTarget);
  $$('#claimDetail .photo-card').forEach(button=>button.onclick=()=>openImage(id,Number(button.dataset.index)));
  showView('claimDetail');
}
function openImage(id,index){
  const claim=claims.find(c=>c.id===id),result=claim?.images[index];if(!result)return;
  selectedClaimId=id;
  const highest=highestModelSignal(result),complete=isComplete(result)&&Boolean(highest),limit=resultThreshold(result),detailScore=highest?.score,review=complete&&detailScore>=limit;
  const comparison=complete?`${percent(detailScore)} ${review?'≥':'<'} ${percent(limit)} · ${review?'검토대상':'정상추정'}`:'분석 미완료 · threshold 비교 및 최종 판정 보류';
  $('#imageDetail').innerHTML=`<div class="detail-head"><div><p class="eyebrow">IMAGE ANALYSIS DETAIL</p><h1>${escapeHtml(result.name)}</h1><p>${escapeHtml(id)} · ${index+1}/${claim.images.length} 이미지</p></div><span class="analysis-signal ${complete?review?'review':'normal':'incomplete'}">${complete?review?'검토대상':'정상추정':'분석미완료'}</span></div>${claimInformationMarkup(claim)}<div class="saved-image-grid"><article class="visual-card"><div class="card-head"><h3>원본 이미지</h3></div><div class="image-stage">${imageMarkup(result.url,'원본 '+result.name)}</div></article><article class="visual-card"><div class="card-head"><div><h3>의심 영역</h3><p>FUSED Heatmap</p></div></div><div class="image-stage">${imageMarkup(result.artifacts?.heatmap,'의심 영역 '+result.name)}</div></article></div><div class="saved-score-grid"><article class="score-card"><h3>모델별 탐지 신호</h3>${modelScoresMarkup(result)}</article><article class="decision-card"><h3>통합 분석 결과</h3><div class="decision-line"><span>통합 점수<small>${highest?`최고 신호 · ${escapeHtml(highest.name)}`:''}</small></span><b class="risk ${complete?tone(detailScore):'medium'}">${percent(complete?detailScore:null)}</b></div><div class="threshold-line"><span>분석 당시 Threshold</span><b>${percent(limit)}</b></div><div class="saved-comparison"><b>${comparison}</b><p>${complete?review?'임계값 이상으로 원본과 추가 증빙 검토가 필요합니다.':'임계값 미만으로 정상으로 추정됩니다.':'실패한 모델이 있어 최종 판단을 내리지 않습니다.'}</p></div></article></div>`;
  $('#backImageClaimBtn').onclick=()=>openClaim(id);
  showView('imageDetail');
}
function roiCases(){
  return claims.map(claim=>{
    const complete=claim.images.filter(isComplete);
    if(!complete.length)return null;
    const representative=complete.reduce((highest,result)=>resultRiskScore(result)>resultRiskScore(highest)?result:highest);
    const amount=Number(claim.amount);
    return {id:claim.id,name:representative.name,score:resultRiskScore(representative),amount:Number.isFinite(amount)&&amount>0?amount:null};
  }).filter(Boolean);
}
function automaticClaimAverage(cases){const amounts=cases.map(item=>item.amount).filter(Number.isFinite);return amounts.length?amounts.reduce((sum,amount)=>sum+amount,0)/amounts.length:null}
async function openRoiSettings(){
  await claimsReady;
  roiCandidate=null;$('#roiApplyBtn').disabled=true;$('#roiSummary').classList.remove('show');
  const [settings,labels]=await Promise.all([
    claimStore.getMeta('roi-settings').catch(()=>null),claimStore.getMeta('roi-labels').catch(()=>({}))
  ]);
  if(settings){$('#roiReviewCost').value=settings.reviewCost??10000;$('#roiFraudLossRate').value=(settings.fraudLossRate??1)*100;$('#roiManualAverage').value=settings.manualAverageAmount??''}
  $('#roiCurrentThreshold').textContent=percent(threshold);
  $('#roiDirectThreshold').value=(threshold*100).toFixed(1);
  const cases=roiCases(),savedLabels=labels||{};
  const automaticAverage=automaticClaimAverage(cases);
  roiManualAverageMode=settings?.averageAmountMode==='manual';
  updateRoiAverageMode(automaticAverage);
  $('#roiRows').innerHTML=cases.map(item=>`<tr><td>${escapeHtml(item.id)}</td><td>${escapeHtml(item.name)}</td><td>${percent(item.score)}</td><td><select class="roi-label" data-claim-id="${escapeHtml(item.id)}" aria-label="${escapeHtml(item.id)} 실제 조작 여부"><option value="">선택 필요</option><option value="0" ${savedLabels[item.id]===0?'selected':''}>0 · 정상 확정</option><option value="1" ${savedLabels[item.id]===1?'selected':''}>1 · 조작 확정</option></select></td></tr>`).join('')||'<tr><td colspan="4" class="empty-state">완료된 분석 결과가 있는 청구건이 없습니다.</td></tr>';
  $$('.roi-label').forEach(select=>select.onchange=invalidateRoiCandidate);
  $('#roiReviewCost').oninput=invalidateRoiCandidate;$('#roiFraudLossRate').oninput=invalidateRoiCandidate;$('#roiManualAverage').oninput=()=>{invalidateRoiCandidate();updateRoiAverageDisplay(automaticAverage)};
  showView('roiSettings');
}
function updateRoiAverageDisplay(automaticAverage){
  const manual=Number($('#roiManualAverage').value),value=roiManualAverageMode&&manual>0?manual:automaticAverage;
  $('#roiAverageAmount').textContent=Number.isFinite(value)?won(value):'—';
  $('#roiAverageSource').textContent=roiManualAverageMode?'사용자 직접 입력값':'저장된 유효 청구건에서 자동 계산';
}
function updateRoiAverageMode(automaticAverage){
  $('#roiManualAverageLabel').classList.toggle('hidden',!roiManualAverageMode);
  $('#roiAverageModeBtn').textContent=roiManualAverageMode?'자동 계산 평균 사용하기':'평균 청구액 직접 입력하기';
  updateRoiAverageDisplay(automaticAverage);
}
$('#roiAverageModeBtn').onclick=()=>{
  roiManualAverageMode=!roiManualAverageMode;invalidateRoiCandidate();
  const cases=roiCases(),automaticAverage=automaticClaimAverage(cases);
  updateRoiAverageMode(automaticAverage);
  if(roiManualAverageMode)$('#roiManualAverage').focus();
};
function invalidateRoiCandidate(){roiCandidate=null;$('#roiApplyBtn').disabled=true;$('#roiSummary').classList.remove('show')}
function calculateRoiThreshold(){
  const cases=roiCases(),reviewCost=Number($('#roiReviewCost').value),fraudLossRate=Number($('#roiFraudLossRate').value)/100;
  if(!cases.length){toast('먼저 분석 결과를 청구건에 저장해 주세요.');return}
  if(!Number.isFinite(reviewCost)||reviewCost<0){toast('검토비용은 0원 이상으로 입력해 주세요.');return}
  if(!Number.isFinite(fraudLossRate)||fraudLossRate<0||fraudLossRate>1){toast('예상손실 비율은 0~100%로 입력해 주세요.');return}
  const labels={};for(const select of $$('.roi-label')){if(select.value===''){toast('모든 청구건의 실제 조작 여부를 선택해 주세요.');select.focus();return}labels[select.dataset.claimId]=Number(select.value)}
  const values=Object.values(labels);if(!values.includes(0)||!values.includes(1)){toast('정상 확정(0)과 조작 확정(1) 사례가 각각 하나 이상 필요합니다.');return}
  const automaticAverage=automaticClaimAverage(cases);
  const manualAverage=Number($('#roiManualAverage').value);
  if(!roiManualAverageMode&&!Number.isFinite(automaticAverage)){toast('청구 금액 입력이 없어 평균 청구액을 직접 입력해 주세요.');roiManualAverageMode=true;updateRoiAverageMode(automaticAverage);$('#roiManualAverage').focus();return}
  if(roiManualAverageMode&&(!Number.isFinite(manualAverage)||manualAverage<=0)){toast('평균 청구액을 0원보다 크게 입력해 주세요.');$('#roiManualAverage').focus();return}
  const averageAmount=roiManualAverageMode?manualAverage:automaticAverage;
  const missedFraudLoss=averageAmount*fraudLossRate;
  let best=null;
  for(let step=0;step<=100;step++){
    const candidate=step/100;let cost=0,reviewed=0,missed=0;
    for(const item of cases){
      if(item.score>=candidate){cost+=reviewCost;reviewed++}
      else if(labels[item.id]===1){cost+=missedFraudLoss;missed++}
    }
    if(!best||cost<best.cost||(cost===best.cost&&reviewed<best.reviewed))best={threshold:candidate,cost,reviewed,missed};
  }
  const baseline=values.filter(value=>value===1).length*missedFraudLoss;
  roiCandidate={...best,reviewCost,fraudLossRate,averageAmount,averageAmountMode:roiManualAverageMode?'manual':'automatic',manualAverageAmount:roiManualAverageMode?manualAverage:null,labels,baseline,savings:baseline-best.cost};
  $('#roiSummary').innerHTML=`<div><span>권장 ROI 임계값</span><b>${percent(best.threshold)}</b></div><div><span>평균 청구액</span><b>${won(averageAmount)}</b></div><div><span>예상 총비용</span><b>${won(best.cost)}</b></div><div><span>미검토 대비 절감액</span><b>${won(roiCandidate.savings)}</b></div>`;
  $('#roiSummary').classList.add('show');$('#roiApplyBtn').disabled=false;
}
async function applyRoiThreshold(){
  if(!roiCandidate||$('#roiApplyBtn').disabled)return;
  $('#roiApplyBtn').disabled=true;
  const settings={threshold:roiCandidate.threshold,reviewCost:roiCandidate.reviewCost,fraudLossRate:roiCandidate.fraudLossRate,averageAmount:roiCandidate.averageAmount,averageAmountMode:roiCandidate.averageAmountMode,manualAverageAmount:roiCandidate.manualAverageAmount,updatedAt:new Date().toISOString()};
  try{await Promise.all([claimStore.setMeta('roi-settings',settings),claimStore.setMeta('roi-labels',roiCandidate.labels)])}
  catch(error){$('#roiApplyBtn').disabled=false;toast('ROI 설정을 저장하지 못했습니다.');return}
  storedRoiSettings=settings;threshold=settings.threshold;updateThresholdUi();
  sendEvent({id:eventId(),action:'set_roi_threshold',threshold});
  toast(`ROI 임계값 ${percent(threshold)}를 적용했습니다.`);showView('claims');
}
async function applyDirectThreshold(){
  const button=$('#roiDirectApplyBtn'),percentage=Number($('#roiDirectThreshold').value);
  if(!Number.isFinite(percentage)||percentage<0||percentage>100){toast('임계값은 0~100% 사이로 입력해 주세요.');$('#roiDirectThreshold').focus();return}
  button.disabled=true;
  const currentSettings=await claimStore.getMeta('roi-settings').catch(()=>null);
  const settings={...(currentSettings||{}),threshold:percentage/100,updatedAt:new Date().toISOString()};
  try{await claimStore.setMeta('roi-settings',settings)}
  catch(error){button.disabled=false;toast('임계값을 저장하지 못했습니다.');return}
  storedRoiSettings=settings;threshold=settings.threshold;updateThresholdUi();
  $('#roiDirectThreshold').value=(threshold*100).toFixed(1);
  sendEvent({id:eventId(),action:'set_roi_threshold',threshold});
  button.disabled=false;toast(`현재 임계값을 ${percent(threshold)}로 변경했습니다.`);
}
$('#roiCalculateBtn').onclick=calculateRoiThreshold;$('#roiApplyBtn').onclick=applyRoiThreshold;$('#roiDirectApplyBtn').onclick=applyDirectThreshold;
function toast(msg){const t=$('#toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2400)}
renderClaims();

setStep(1);

$('.system-state').innerHTML='<i></i>분석 서버 연결 대기';
sendComponent('streamlit:componentReady',{apiVersion:1});
let resizeFrame=null;
new ResizeObserver(()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>sendComponent('streamlit:setFrameHeight',{height:Math.ceil($('main').getBoundingClientRect().bottom+32)}))}).observe($('main'));
