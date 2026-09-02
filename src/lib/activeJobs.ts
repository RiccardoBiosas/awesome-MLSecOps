export type ExpiringJob = {
  data: {
    expires_date: Date;
  };
};

function utcDateValue(value: Date): number {
  return Date.UTC(value.getUTCFullYear(), value.getUTCMonth(), value.getUTCDate());
}

export function activeJobs<T extends ExpiringJob>(jobs: T[], now = new Date()): T[] {
  const today = utcDateValue(now);
  return jobs.filter((job) => utcDateValue(job.data.expires_date) >= today);
}