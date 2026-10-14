// Groq provider: request shape, response/error handling, rate-limit pacing, and a full
// pipeline run with the Groq profile (scripted model, fake services).
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { createEnv, makeGmail, makeUrlFetch, groqOk, FakeSpreadsheet } = require('./harness');
const { F, C, mail, scriptedModel, ME } = require('./fixtures');

const KEY = 'gsk_test0000000000000000000000000000';
const plain = v => JSON.parse(JSON.stringify(v));

function providerEnv(responses) {
  const env = createEnv({ config: { USER_TIMEZONE: 'Asia/Kolkata' } });   // default provider: groq
  env.props.set('GROQ_API_KEY', KEY);
  const f = makeUrlFetch(() => responses.shift());
  env.ctx.UrlFetchApp = f;
  return { env, f };
}
const req = { model: 'openai/gpt-oss-120b', maxTokens: 2500, system: 'sys', user: 'u', schema: { type: 'object' } };
const err = (status, message, code, headers) => ({ status, headers: headers || {}, body: { error: { message, type: 'invalid_request_error', code } } });

test('Groq is the default provider and its profile is used', () => {
  const env = createEnv();
  assert.equal(env.x.CONFIG.LLM.PROVIDER, 'groq');
  assert.equal(env.run('llm_').MODEL, 'openai/gpt-oss-120b');
  assert.equal(env.run('llm_').TRIAGE_MODEL, 'openai/gpt-oss-20b');
});

test('request: OpenAI-compatible body with strict JSON schema, bearer key', () => {
  const { env, f } = providerEnv([groqOk({ ok: true })]);
  const out = env.x.LLM.run([req]);
  assert.equal(out[0].ok, true);
  const call = f.calls[0];
  assert.equal(call.url, 'https://api.groq.com/openai/v1/chat/completions');
  assert.equal(call.headers.Authorization, 'Bearer ' + KEY);
  assert.deepEqual(plain(call.body.messages.map(m => m.role)), ['system', 'user']);
  assert.deepEqual(plain(call.body.response_format), { type: 'json_schema', json_schema: { name: 'result', strict: true, schema: { type: 'object' } } });
  assert.equal(call.body.reasoning_effort, 'low');
  assert.equal(call.body.include_reasoning, false);
  assert.equal(call.body.max_completion_tokens, 2500);
  assert.equal(out[0].costUsd, 0, 'free tier priced at 0');
  assert.ok(!JSON.stringify(out).includes(KEY));
});

test('errors are categorised', () => {
  const { env } = providerEnv([
    err(400, 'Generated JSON does not match the expected schema. Please adjust your prompt.', 'json_validate_failed'),
    err(401, 'Invalid API Key', 'invalid_api_key'),
    err(413, 'Request too large for model', 'request_too_large'),
    err(503, 'Service Unavailable'),
    { status: 200, body: { model: 'openai/gpt-oss-120b', choices: [{ finish_reason: 'length', message: { content: '{"a":' } }], usage: {} } },
    groqOk('{not json'),
    err(429, 'Rate limit reached', 'rate_limit_exceeded', { 'retry-after': '120' })
  ]);
  const out = env.x.LLM.run([req, req, req, req, req, req, req]);
  assert.deepEqual(plain(out.map(o => o.ok ? 'ok' : o.error.category)), ['invalid_output', 'config', 'permanent', 'transient', 'invalid_output', 'invalid_output', 'rate_limit']);
  assert.equal(out[6].error.retryAfterSec, 120, 'long waits go back to the queue');
  assert.ok(out.every(o => !o.error || !o.error.dailyLimit), 'only a daily token limit is marked dailyLimit');
  assert.deepEqual(plain(env.sleeps), []);
});

// Groq's 429 messages as returned by the API (organization ID replaced).
const tpd = model => 'Rate limit reached for model `' + model + '` in organization `org_test` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 197978, Requested 2670. Please try again in 4m39.936s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing';
const tpm = model => 'Rate limit reached for model `' + model + '` in organization `org_test` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 7000, Requested 2500. Please try again in 9.5s.';

test('a daily token limit is recognised; a per-minute limit is not', () => {
  const { env } = providerEnv([
    err(429, tpd('openai/gpt-oss-120b'), 'rate_limit_exceeded', { 'retry-after': '280' }),
    err(429, tpm('openai/gpt-oss-120b'), 'rate_limit_exceeded', { 'retry-after': '60' })
  ]);
  const out = env.x.LLM.run([req, req]);
  assert.deepEqual(plain(out.map(o => [o.error.category, !!o.error.dailyLimit])), [['rate_limit', true], ['rate_limit', false]]);
  assert.equal(out[0].error.retryAfterSec, 280);
});

test('short 429 waits are slept through once; nearly exhausted token window is waited out', () => {
  const { env, f } = providerEnv([
    err(429, 'Rate limit reached', 'rate_limit_exceeded', { 'retry-after': '7' }),
    groqOk({ ok: 1 }, null, { 'x-ratelimit-remaining-tokens': '1200', 'x-ratelimit-reset-tokens': '6.5s' }),
    groqOk({ ok: 2 })
  ]);
  const out = env.x.LLM.run([req, req]);
  assert.deepEqual(plain(out.map(o => o.ok)), [true, true]);
  assert.equal(f.calls.length, 3);
  assert.deepEqual(plain(env.sleeps), [7000, 6500]);
});

test('no Groq key: config error, no request', () => {
  const { env, f } = providerEnv([]);
  env.props.delete('GROQ_API_KEY');
  assert.equal(env.x.LLM.run([req])[0].error.category, 'config');
  assert.equal(f.calls.length, 0);
  assert.equal(env.x.LLM.hasKey(), false);
});

test('Set LLM API key validates and stores a Groq key under GROQ_API_KEY', () => {
  const env = createEnv({ prompts: ['sk-ant-wrong-provider-key-000000000000'] });
  env.run('setApiKey');
  assert.equal(env.props.has('GROQ_API_KEY'), false);
  const env2 = createEnv({ props: env.props, prompts: [KEY] });
  env2.run('setApiKey');
  assert.equal(env.props.get('GROQ_API_KEY'), KEY);
});

test('prompts are trimmed to the Groq profile budget', () => {
  const env = createEnv({ config: { USER_TIMEZONE: 'Asia/Kolkata' } });
  const long = 'x'.repeat(20000);
  const email = { messageId: 'm', threadId: 't', date: new Date('2026-03-09T05:30:00Z'), direction: 'received', sender: { name: '', email: 'a@b.example', domain: 'b.example' },
    to: '', subject: 's', current: long, quoted: long, forwarded: false, labels: [] };
  const r = env.x.Semantic.buildClassifyRequest(email, [], [], []);
  assert.ok(r.user.length < 2500 + 500 + 1500, 'user prompt ' + r.user.length + ' chars');
  assert.equal(r.model, 'openai/gpt-oss-120b');
});

test('input limits: 2,500 chars of new text, 500 of quoted text; subject, sender and date are never cut', () => {
  const env = createEnv({ config: { USER_TIMEZONE: 'Asia/Kolkata' } });
  const current = 'Your application for Firmware Intern (Job ID R-55012) at Honeywell was received. ' + 'a'.repeat(2500) + 'AFTER_LIMIT';
  const quoted = 'Q'.repeat(500) + 'QUOTED_AFTER_LIMIT';
  const email = { messageId: 'm', threadId: 't', date: new Date('2026-10-13T05:30:00Z'), direction: 'received',
    sender: { name: 'Honeywell', email: 'honeywell@myworkday.com', domain: 'myworkday.com' }, to: '', subject: 'Honeywell - Firmware Intern R-55012 application received',
    current: current, quoted: quoted, forwarded: false, labels: [] };
  const u = env.x.Semantic.buildClassifyRequest(email, [], [], []).user;
  const tag = name => u.match(new RegExp('<' + name + '>\\n([\\s\\S]*?)\\n</' + name + '>'))[1];
  assert.equal(tag('current_message'), current.slice(0, 2500));
  assert.equal(tag('quoted_history'), quoted.slice(0, 500));
  assert.match(tag('current_message'), /Firmware Intern \(Job ID R-55012\) at Honeywell/, 'the opening of the new text is kept');
  const meta = JSON.parse(tag('email_metadata'));
  assert.equal(meta.subject, email.subject);
  assert.match(meta.from, /honeywell@myworkday\.com/);
  assert.match(meta.sent_at, /^2026-10-13T11:00:00\+05:30/);
});

test('input limits: only the one most recent earlier thread message is sent as context', () => {
  const thread = [1, 2, 3].map(i => mail('t-ctx', '2026-10-1' + i + 'T05:00:00Z', 'Recruiter <r@honeywell.example>', 'Re: Firmware Intern', 'message number ' + i, C({})));
  const env = createEnv({ gmail: makeGmail(thread, ME), config: { USER_TIMEZONE: 'Asia/Kolkata' } });
  const fc = env.run('fetchWithContext_', thread[2].id, 't-ctx', ME);
  assert.equal(fc.context.length, 1);
  assert.match(fc.context[0].text, /message number 2/);
  assert.match(fc.email.current, /message number 3/);
});

test('full pipeline with the Groq profile', () => {
  const mailbox = [F.acmeConfirm(), F.acmeDataConfirm(), F.acmeInterviewIndirect(), F.acmeScheduledNoKeywords(), F.acmeOffer(),
    F.globexUnusual(), F.globexRejection(), F.jobAlert(), F.quotedRejectionMisread()];
  const ss = new FakeSpreadsheet();
  const props = new Map([['GROQ_API_KEY', KEY]]);
  const gmail = makeGmail(mailbox, ME);
  const fetch = makeUrlFetch(scriptedModel(mailbox, createEnv().x));
  const mk = () => createEnv({ ss, props, gmail, fetch, config: { USER_TIMEZONE: 'Asia/Kolkata', NUMERIC_DATE_ORDER: '', DISCOVERY: { BACKFILL_START: '2026-01-01', PAGE_SIZE: 100 } } });
  mk().run('setupWorkbook_', ss);
  mk().run('startBackfill');
  const apps = ss.tab('Applications').records();
  const st = (c, t) => apps.find(a => a.Company === c && a['Job Title'] === t)['Current Status'];
  assert.equal(apps.length, 3);
  assert.equal(st('Acme Robotics', 'Software Engineer Intern'), 'Offer');
  assert.equal(st('Acme Robotics', 'Data Analyst Intern'), 'Applied');
  assert.equal(st('Globex', 'Backend Developer (Graduate)'), 'Rejected');
  assert.ok(fetch.calls.every(c => c.body.response_format && /gpt-oss/.test(c.body.model)));
  assert.ok(fetch.calls.filter(c => !c.body.response_format.json_schema.schema.properties.results).every(c => c.body.model === 'openai/gpt-oss-120b'), 'primary model classifies; no fallback');
  const triage = fetch.calls.filter(c => c.body.response_format.json_schema.schema.properties.results);
  assert.ok(triage.every(c => c.body.model === 'openai/gpt-oss-20b'));
  assert.ok(triage.every(c => JSON.parse(c.body.messages[1].content.match(/<messages>\n([\s\S]*)\n<\/messages>/)[1]).length <= 10), 'triage batch of 10');
  assert.ok(ss.tab('Review Queue').records().some(r => /No supporting quote/.test(r['Review Reason'])));
  const log = ss.tab('Sync Log').records().slice(-1)[0];
  assert.equal(log['Est. Cost USD'], 0);
});

// ---------- fallback to the second model when the primary model's daily token limit is reached ----------

const PRIMARY = 'openai/gpt-oss-120b', FALLBACK = 'openai/gpt-oss-20b';
const isClassify = c => !c.body.response_format.json_schema.schema.properties.results;

/** Groq pipeline over `mailbox`; `answer(call, scripted)` decides each classification response. */
function fallbackWorld(mailbox, answer) {
  const ss = new FakeSpreadsheet();
  const scripted = scriptedModel(mailbox, createEnv().x);
  const fetch = makeUrlFetch(call => (isClassify(call) ? answer(call, scripted) : scripted(call)));
  const props = new Map([['GROQ_API_KEY', KEY]]);
  const gmail = makeGmail(mailbox, ME);
  const mk = () => createEnv({ ss, props, gmail, fetch, config: { USER_TIMEZONE: 'Asia/Kolkata', NUMERIC_DATE_ORDER: '', DISCOVERY: { BACKFILL_START: '2026-01-01' } } });
  mk().run('setupWorkbook_', ss);
  return { ss, fetch, run: () => mk().run('startBackfill'), classifyCalls: () => fetch.calls.filter(isClassify),
    tab: n => ss.tab(n).records(), log: () => ss.tab('Sync Log').records().slice(-1)[0] };
}
const limited = (model, msg) => err(429, (msg || tpd)(model), 'rate_limit_exceeded', { 'retry-after': '280' });

test('fallback: primary daily limit -> one retry on the fallback model, normal pipeline, rest of the run on the fallback', () => {
  const w = fallbackWorld([F.acmeConfirm(), F.globexUnusual()], (call, scripted) => {
    if (call.body.model === PRIMARY) return limited(PRIMARY);
    const r = scripted(call);
    r.body.model = FALLBACK;
    return r;
  });
  w.run();
  assert.deepEqual(plain(w.classifyCalls().map(c => c.body.model)), [PRIMARY, FALLBACK, FALLBACK], 'one primary attempt, then fallback only');
  const apps = w.tab('Applications');
  assert.equal(apps.length, 2, 'fallback results applied through the normal pipeline');
  assert.ok(w.tab('Email Events').every(e => e.Model === FALLBACK), 'model recorded per event');
  assert.ok(w.tab('Work Queue').every(t => t.State === 'done'));
  assert.match(w.log().Notes, /gpt-oss-120b daily token limit reached; classifying with openai\/gpt-oss-20b/);
  assert.ok(!/Hi Sam|received your application/.test(w.log().Notes), 'no email text in the log');
});

test('fallback: both models at their daily limit -> stops, nothing lost, no attempts used, no loop', () => {
  const w = fallbackWorld([F.acmeConfirm(), F.globexUnusual()], call => limited(call.body.model));
  w.run();
  assert.deepEqual(plain(w.classifyCalls().map(c => c.body.model)), [PRIMARY, FALLBACK], 'exactly one fallback attempt, then the run stops');
  const q = w.tab('Work Queue');
  assert.ok(q.every(t => Number(t.Attempts || 0) === 0 && (t.State === 'pending' || t.State === 'retry')), JSON.stringify(q));
  const tried = q.filter(t => t.State === 'retry');
  assert.equal(tried.length, 1, 'the other message is left untouched for the next run');
  assert.equal(tried[0]['Error Category'], 'rate_limit');
  assert.ok(tried[0]['Next Attempt At'] > new Date(Date.now() + 270e3), 'waits for the provider retry-after');
  assert.equal(w.tab('Applications').length, 0);
  assert.equal(w.tab('Review Queue').length, 0, 'not dead: retried later');
});

test('fallback is not used for auth errors, invalid requests, per-minute limits or server errors', () => {
  const cases = [
    [err(401, 'Invalid API Key', 'invalid_api_key'), 'config'],
    [err(400, "'messages' must contain the word 'json'", 'invalid_request_error'), 'config'],
    [limited(PRIMARY, tpm), 'rate_limit'],
    [err(503, 'Service Unavailable'), 'transient']
  ];
  for (const [response, category] of cases) {
    const w = fallbackWorld([F.acmeConfirm()], () => response);
    w.run();
    assert.ok(w.classifyCalls().every(c => c.body.model === PRIMARY), category + ': no fallback call');
    assert.equal(w.tab('Work Queue')[0]['Error Category'], category);
  }
});

test('fallback results get the same checks: low confidence -> Review Queue; malformed -> retried, never applied', () => {
  const low = fallbackWorld([F.acmeConfirm()], (call, scripted) => {
    if (call.body.model === PRIMARY) return limited(PRIMARY);
    const r = scripted(call);
    const cls = JSON.parse(r.body.choices[0].message.content);
    cls.confidence = 'low';
    return groqOk(cls, FALLBACK);
  });
  low.run();
  assert.equal(low.tab('Applications').length, 0);
  assert.match(low.tab('Review Queue')[0]['Review Reason'], /confidence is low/);

  const bad = fallbackWorld([F.acmeConfirm()], call => (call.body.model === PRIMARY ? limited(PRIMARY) : groqOk('{not json', FALLBACK)));
  bad.run();
  const t = bad.tab('Work Queue')[0];
  assert.equal(t.State, 'retry');
  assert.equal(t['Error Category'], 'invalid_output');
  assert.equal(bad.tab('Applications').length, 0);
});
