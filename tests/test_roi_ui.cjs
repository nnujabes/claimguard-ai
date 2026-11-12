/* Browser regression test for ROI threshold settings. */
const assert = require('assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const result = (name, score) => ({name, complete:true, score, threshold:.63, route:score>=.63?'HUMAN_REVIEW':'AOS', scores:{trufor:score,fused:score,recapture:score}, models:{trufor:{status:'complete',score},fused:{status:'complete',score},recapture:{status:'complete',score}}, artifacts:{}});

(async()=>{
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    async function load(){
      await page.goto(process.env.CLAIMGUARD_URL||'http://127.0.0.1:8517');
      await page.waitForSelector('iframe');
      const frame=await (await page.$('iframe')).contentFrame();
      await frame.waitForFunction(()=>document.querySelector('.system-state')?.textContent.includes('연결됨'));
      await frame.evaluate(()=>claimsReady);
      return frame;
    }
    let frame=await load();
    await frame.evaluate(async cases=>{
      const records=[
        {id:'ROI-A',date:'2026-09-18',amount:1000000,status:'심사대기',images:cases.a},
        {id:'ROI-B',date:'2026-09-18',amount:3000000,status:'심사완료',images:cases.b},
      ];
      for(const record of records)await claimStore.put(record);
      claims=records;
    },{a:[result('a-high.png',.8),result('a-low.png',.2)],b:[result('b.png',.3)]});
    await frame.click('#roiSettingsBtn');
    await frame.waitForSelector('#roiSettingsView.active');
    assert.equal(await frame.$eval('#roiCurrentThreshold',el=>el.textContent),'63.0%');
    await frame.fill('#roiDirectThreshold','47.5');
    await frame.click('#roiDirectApplyBtn');
    await frame.waitForFunction(()=>Math.abs(threshold-.475)<.000001);
    assert.equal(await frame.$eval('#roiCurrentThreshold',el=>el.textContent),'47.5%');
    assert.equal(await frame.evaluate(async()=>(await claimStore.getMeta('roi-settings')).threshold),.475,'direct threshold persists');
    assert.equal(await frame.$$eval('#roiRows tr',rows=>rows.length),2,'one ROI row per claim');
    assert.match(await frame.$eval('#roiRows',el=>el.textContent),/80.0%/,'highest image score represents the claim');
    await frame.selectOption('[data-claim-id="ROI-A"]','1');
    await frame.selectOption('[data-claim-id="ROI-B"]','0');
    await frame.fill('#roiReviewCost','10000');
    await frame.fill('#roiFraudLossRate','50');
    await frame.click('#roiCalculateBtn');
    const summary=await frame.$eval('#roiSummary',el=>el.textContent);
    for(const text of ['31.0%','2,000,000원','10,000원','990,000원'])assert.ok(summary.includes(text),text);
    await frame.click('#roiAverageModeBtn');
    await frame.fill('#roiManualAverage','4000000');
    await frame.click('#roiCalculateBtn');
    const manualSummary=await frame.$eval('#roiSummary',el=>el.textContent);
    for(const text of ['31.0%','4,000,000원','10,000원','1,990,000원'])assert.ok(manualSummary.includes(text),text);
    await frame.click('#roiApplyBtn');
    await frame.waitForFunction(()=>Math.abs(threshold-.31)<.000001);
    assert.equal(await frame.$eval('#roiSettingsBtn strong',el=>el.textContent),'31%');
    frame=await load();
    await frame.waitForFunction(()=>Math.abs(threshold-.31)<.000001);
    assert.equal(await frame.$eval('#roiSettingsBtn strong',el=>el.textContent),'31%','threshold persists after reload');
    await frame.click('#roiSettingsBtn');
    await frame.waitForSelector('#roiSettingsView.active');
    assert.equal(await frame.$eval('#roiManualAverage',el=>el.value),'4000000','manual average persists');
    assert.equal(await frame.$eval('#roiManualAverageLabel',el=>!el.classList.contains('hidden')),true);
    assert.deepEqual(errors,[]);
    console.log('PASS: claim-level max score, automatic/manual average claim amount, ROI optimization, apply and reload persistence');
  }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exit(1)});
