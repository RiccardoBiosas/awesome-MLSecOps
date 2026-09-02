import { describe, expect, it } from "vitest";
import { activeJobs } from "./activeJobs";

const job = (expires: string) => ({ data: { expires_date: new Date(`${expires}T00:00:00Z`) } });

describe("activeJobs", () => {
  it("keeps listings through the expiration date", () => {
    const jobs = [job("2026-07-19")];
    expect(activeJobs(jobs, new Date("2026-07-19T23:59:59Z"))).toHaveLength(1);
  });

  it("removes listings after the expiration date", () => {
    const jobs = [job("2026-07-18"), job("2026-07-20")];
    expect(activeJobs(jobs, new Date("2026-07-19T08:00:00Z"))).toEqual([jobs[1]]);
  });
});