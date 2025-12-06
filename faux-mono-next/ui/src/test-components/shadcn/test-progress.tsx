import { Progress } from "@bluevoid/ui/progress";
import {  useState } from "react";

function TestProgress() {
  const [progress, setProgress] = useState(13);
	const [progress2, setProgress2] = useState(0);

	return (
	<div className="flex flex-col gap-6 w-full">
		<Progress
			value={progress}
			className="w-[60%]"
			onClick={() => setProgress((prev) => (prev - 20 > 0 ? prev - 20 : 100))}
		/>
		<Progress
			value={progress2}
			className="w-[60%]"
			onClick={() => setProgress2((prev) => (prev + 10 <= 100 ? prev + 10 : 0))}
		/>
	</div>
	);
}

export default TestProgress;
