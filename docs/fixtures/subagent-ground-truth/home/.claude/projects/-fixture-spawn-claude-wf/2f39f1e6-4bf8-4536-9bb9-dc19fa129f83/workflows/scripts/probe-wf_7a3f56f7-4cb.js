export const meta = { name: 'probe', description: 'one agent probe', phases: [{ title: 'Probe' }] };
const r = await agent('Reply with the single word OK and stop.', { label: 'zeta', phase: 'Probe' });
return { r };